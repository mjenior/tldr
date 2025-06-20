"""
This module contains the core logic for generating TLDR summaries.

It provides a class, TldrEngine, which is responsible for processing input files,
fetching additional context, optionally performing Retrieval Augmented Generation (RAG)
using local embeddings, refining user queries, and generating summaries based on the
provided information. It manages interactions with a completion API and handles the
output of intermediate and final results.
"""

import asyncio
from .openai import CompletionHandler


class TldrEngine(CompletionHandler):
    """
    Handles the core logic for generating TLDR summaries.

    This class is responsible for processing input files, fetching additional context,
    optionally performing Retrieval Augmented Generation (RAG) using local embeddings,
    refining user queries, and generating summaries based on the provided information.
    It manages interactions with a completion API and handles the output of
    intermediate and final results.

    Attributes:
        recursive (bool): Whether to search for input files recursively in directories.
        verbose (bool): If True, enables detailed logging.
        api_key (str, optional): The API key for the completion service.
        context_size (str): Specifies the context window size for the model (e.g., "small", "medium", "large").
        tag (str): A tag for the current run, used for naming output directories.
        output_directory (str): The path to the directory where intermediate and final files are saved.
        content (list[str]): A list of strings, where each string is the content of an input file.
        query (str): The user's query to be addressed by the summary.
        added_context (str): Formatted additional context to be used in prompts.
    """

    def __init__(
        self,
        verbose=True,
        api_key=None,
    ):
        # Initialize parent class chain
        super().__init__(api_key=api_key)

        # Additional attributes
        self.verbose = verbose
        self.random_seed = 42
        self.content = None
        self.query = None
        self.added_context = ""
        self.platform = "openai"

        # Initialize session
        self._generate_run_tag()
        self._create_output_path()
        self._start_logging()
        self._read_system_instructions()
        self._new_openai_client()

    async def load_all_content(self, input_files=None, context_files=None, context_size="medium", recursive=False):
        """Establish context for the TldrEngine"""

        # Read in files and contents
        if input_files is not None:
            self.content = self.fetch_content(user_files=input_files, label="reference")
        else:
            self.content = self.fetch_content(recursive=recursive, label="reference")

        # Fetch and reformat additional context if provided
        if context_files is not None:
            all_context = self.fetch_content(user_files=context_files, label="context")
            await self.format_context("".join(all_context), context_size=context_size, label="context_files")

        # Create embeddings
        await self._create_embeddings(context_size=context_size)

    async def _create_embeddings(self, context_size="medium", platform="openai"):
        """Create local embeddings for reference documents"""

        # Create local embeddings
        self.encode_text_to_vector_store(platform=platform)

        # Search for added initial context
        context_search = await self.search_embedded_context(query=self.query)
        await self.format_context(context_search, context_size=context_size, label="context_search")

    async def finish_session(self):
        """Complete run with stats reporting"""
        result = self.save_response_text(self.added_context, label="full_context")
        self.logger.info(result)
        self.format_spending()
        self.generate_token_report()
        await self.client.close()

    async def format_context(self, new_context, context_size="medium", label="formatted_context"):
        """
        Creates more concise, less repetative context message
        """

        # Format context references
        formatted_context = await self.single_completion(
            message=new_context, prompt_type="format_context", context_size=context_size
        )
        self.added_context += formatted_context["response"]
        result = self.save_response_text(
            formatted_context["response"],
            label=label,
        )
        self.logger.info(result)

    def _lint_query(self, query) -> str:
        """
        Normalizes user query to ensure it's more well-formed & AI-friendly
        """
        # Normalize input to string
        if isinstance(query, list):
            processed_query = " ".join(map(str, query)).strip()
        else:
            processed_query = str(query).strip()

        # Ensure it ends with a suitable punctuation mark
        if processed_query[-1] not in ".!?":
            processed_query += "?"

        return processed_query

    async def refine_user_query(self, query, context_size="medium"):
        """
        Automatically improve user query for greater specificity
        """
        if query is None or query.strip() == "":
            self.query = ""
            return
        elif self.refine_query is False:
            self.query = self._lint_query(query)
            return
        else:
            self.logger.info("Refining user query...")

        # Generate new query text
        refined_query = await self.single_completion(
            message=self._lint_query(query), prompt_type="refine_prompt", context_size=context_size
        )

        # Handle output text
        self.query = refined_query["response"]
        result = self.save_response_text(self.query, label="refined_query")
        self.logger.info(result)

    async def summarize_resources(self):
        """Generate component and synthesis summary text"""
        self.logger.info("Generating summaries for selected resources...")

        # Asynchronously summarize documents
        reference_summaries = await self.multi_completions(self.content)

        # Annotate and save response strings
        self.reference_summaries = []
        for i, summary in enumerate(reference_summaries):
            self.reference_summaries.append(
                f"Reference {i} Summary:\n{summary['response']}\n"
            )
            result = self.save_response_text(
                summary["response"], label="reference_summary", idx=i
            )
            self.logger.info(result)

    async def integrate_summaries(self, context_size="medium"):
        """Generate integrated executive summaries combine all current summary text"""
        self.logger.info("Generating integrated summary text...")

        # Generate executive summary
        executive_summary = await self.single_completion(
            message="\n\n".join(self.reference_summaries), prompt_type="executive_summary", context_size=context_size
        )
        self.executive_summary = executive_summary["response"]

        # Handle output
        result = self.save_response_text(
            executive_summary["response"], label="executive_summary"
        )
        self.logger.info(result)

    async def polish_response(self, tone="stylized", context_size="medium"):
        """Refine final response text"""
        self.logger.info("Finalizing response text...")

        # Select tone
        tone_instructions = (
            "formal_polishing" if tone == "formal" else "stylized_polishing"
        )

        # Polish summary
        self.polished_summary = await self.single_completion(
            message=self.executive_summary,
            prompt_type=tone_instructions,
            context_size=context_size,
        )
        self.polished_summary = self.polished_summary["response"]
        result = self.save_response_text(
            self.polished_summary,
            label="polished_summary",
        )
        self.logger.info(result)

    async def save_to_pdf(self, smart_title=True, polished=True):
        """Saves polished summary string to formatted PDF document."""
        self.logger.info("Saving final summary to PDF...")
        summary_text = self.polished_summary if polished is True else self.executive_summary

        # Generate title
        if smart_title is True:
            doc_title = await self.single_completion(
                message=summary_text, prompt_type="title_instructions", context_size="low"
            )

        # Save to PDF
        pdf_path = self.generate_tldr_pdf(
            summary_text=summary_text,
            doc_title=doc_title["response"],
        )
        self.logger.info(f"Final summary saved to {pdf_path}")

    async def apply_research(self, context_size="medium"):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""        
        self.logger.info("Applying research agent to knowledge gaps...")
        research_questions = await self.single_completion(
            message="\n\n".join(self.reference_summaries), prompt_type="background_gaps", context_size="medium"
        )
        research_questions = research_questions["response"].split("\n")

        # Search web for to fill gaps
        self.research_results = await self.multi_completions(
            research_questions, prompt_type="web_search", context_size=context_size
        )
        research_context = "\n".join([x["response"] for x in self.research_results])

        # Also search for additional context missed in references
        for question in research_questions:
            search_context = self.search_embedded_context(query=question, num_results=2)
            research_context += "\n" + search_context
        
        # Update added context with new research
        await self.format_context(research_context, label="research_context")

        # Handle output
        divider = "#--------------------------------------------------------#"
        joint_text = "\n".join(
            [
                f'Question: {q_ans_pair["prompt"]}\nAnswer: {q_ans_pair["response"]}\n\n{divider}\n\n'
                for q_ans_pair in self.research_results
            ]
        )
        result = self.save_response_text(joint_text, label="research_results")
        self.logger.info(result)
