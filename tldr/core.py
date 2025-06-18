"""
This module contains the core logic for generating TLDR summaries.

It provides a class, TldrEngine, which is responsible for processing input files,
fetching additional context, optionally performing Retrieval Augmented Generation (RAG)
using local embeddings, refining user queries, and generating summaries based on the
provided information. It manages interactions with a completion API and handles the
output of intermediate and final results.
"""

import re
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
        input_files (list[str], optional): A list of file paths to be summarized.
        context_files (list[str], optional): A list of file paths to be used as additional context.
        recursive_search (bool): Whether to search for input files recursively in directories.
        verbose (bool): If True, enables detailed logging.
        api_key (str, optional): The API key for the completion service.
        local_rag (bool): If True, enables local Retrieval Augmented Generation.
        context_size (str): Specifies the context window size for the model (e.g., "small", "medium", "large").
        tag (str): A tag for the current run, used for naming output directories.
        output_directory (str): The path to the directory where intermediate and final files are saved.
        content (list[str]): A list of strings, where each string is the content of an input file.
        raw_context (str, optional): Combined content of all context files.
        query (str): The user's query to be addressed by the summary.
        added_context (str): Formatted additional context to be used in prompts.
    """

    def __init__(
        self,
        input_files=None,
        query=None,
        refine_query=True,
        context_files=None,
        recursive_search=False,
        verbose=True,
        web_search=True,
        context_size="medium",
        tone="stylized",
        polish=False,
        pdf=True,
        testing=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.context_size = context_size
        self.recursive_search = recursive_search
        self.query = query
        self.web_search = web_search
        self.refine_query = refine_query
        self.added_context = ""
        self.polish = polish
        self.tone = tone
        self.pdf = pdf
        self.testing = testing
        self.random_seed = 42
        self.content = None
        self.input_files = input_files
        self.context_files = context_files

        # Check for testing mode
        if self.testing:
            self.web_search = False
            self.query = None
            self.polish = False
            self.context_size = "low"

    async def initialize_async_session(self):
        """Initialize the TldrEngine asynchronously"""

        self._generate_run_tag()
        self._create_output_path()
        self._start_logging()
        self._read_system_instructions()
        self._new_openai_client()
        await self.refine_user_query()
        
        # Read in files and contents
        if self.input_files is not None and len(self.input_files) > 0:
            self.content = self.fetch_content(user_files=self.input_files, label="reference")
        else:
            self.content = self.fetch_content(
                recursive=self.recursive_search, label="reference"
            )

        # Fetch and reformat additional context if provided
        if self.context_files is not None:
            all_context = self.fetch_content(user_files=self.context_files, label="context")
            await self.format_context("".join(all_context), label="context_files")

        # Create embeddings
        await self.create_embeddings()

    async def create_embeddings(self):
        """Create local embeddings for reference documents"""

        # Create local embeddings
        self.encode_text_to_vector_store()

        # Search for added initial context
        context_search = await self.search_embedded_context(query=self.query)
        await self.format_context(context_search, label="context_search")

    async def finish_session(self):
        """Complete run with stats reporting"""
        result = self.save_response_text(self.added_context, label="full_context")
        self.logger.info(result)
        self.format_spending()
        self.generate_token_report()
        await self.client.close()

    async def format_context(self, new_context, label="formatted_context"):
        """Creates more concise, less repetative context message."""

        # Format context references
        formatted_context = await self.single_completion(
            message=new_context, prompt_type="format_context"
        )
        self.added_context += formatted_context["response"]
        result = self.save_response_text(
            formatted_context["response"],
            label=label,
        )
        self.logger.info(result)

    @staticmethod
    def _lint_query(current_query: str | list) -> str:
        """
        Normalizes a user query to ensure it's a well-formed, AI-friendly string.
        """
        # Normalize input to string
        if isinstance(current_query, list):
            processed_query = " ".join(map(str, current_query))
        else:
            processed_query = str(current_query)

        # Strip leading/trailing whitespace and normalize internal whitespace
        processed_query = re.sub(r"\s+", " ", processed_query.strip())

        # Capitalize first letter if not already
        processed_query = processed_query[0].upper() + processed_query[1:]

        # Ensure it ends with a suitable punctuation mark
        if processed_query[-1] not in ".!?":
            processed_query += "?"

        return processed_query

    async def refine_user_query(self):
        """Attempt to automatically improve user query for greater specificity"""
        if self.query is None or self.query.strip() is None or self.query.strip() == "":
            self.query = ""
            return

        self.logger.info("Refining user query...")
        # Check text formatting
        self.query = self._lint_query(self.query)

        # Generate new query text
        refined_query = await self.single_completion(
            message=self.query, prompt_type="refine_prompt"
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

    async def integrate_summaries(self):
        """Generate integrated executive summaries combine all current summary text"""
        self.logger.info("Generating integrated summary text...")

        # Generate executive summary
        executive_summary = await self.single_completion(
            message="\n\n".join(self.reference_summaries), prompt_type="executive_summary"
        )
        self.executive_summary = executive_summary["response"]

        # Handle output
        result = self.save_response_text(
            executive_summary["response"], label="executive_summary"
        )
        self.logger.info(result)

    async def polish_response(self):
        """Refine final response text"""
        self.logger.info("Finalizing response text...")

        # Select tone
        tone_instructions = (
            "formal_polishing" if self.tone == "formal" else "stylized_polishing"
        )

        # Polish summary
        self.polished_summary = await self.single_completion(
            message=self.executive_summary,
            prompt_type=tone_instructions,
        )
        self.polished_summary = self.polished_summary["response"]
        result = self.save_response_text(
            self.polished_summary,
            label="polished_summary",
        )
        self.logger.info(result)

    async def save_to_pdf(self, summary_text):
        """Saves polished summary string to formatted PDF document."""
        self.logger.info("Saving final summary to PDF...")

        # Generate title
        doc_title = await self.single_completion(
            message=summary_text, prompt_type="title_instructions"
        )

        # Save to PDF
        pdf_path = self.generate_tldr_pdf(
            summary_text=summary_text,
            doc_title=doc_title["response"],
        )
        self.logger.info(f"Final summary saved to {pdf_path}")

    async def apply_research(self):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""
        if self.web_search is False:
            return
        
        self.logger.info("Applying research agent to knowledge gaps...")
        research_questions = await self.single_completion(
            message="\n\n".join(self.reference_summaries), prompt_type="background_gaps"
        )
        research_questions = research_questions["response"].split("\n")

        # Search web for to fill gaps
        self.research_results = await self.multi_completions(
            research_questions, prompt_type="web_search"
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
