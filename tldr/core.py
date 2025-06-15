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
        self.tone = tone
        self.raw_context = None
        self.polish = polish
        self.pdf = pdf
        self.testing = testing
        self.random_seed = 42

        # Set up logger and intermediate file output directory
        self.setup_logging()
        self._create_output_path()

        # Read in system instructions
        self.read_system_instructions()

        # Read in files and contents
        if input_files is not None:
            self.content = self.fetch_content(user_files=input_files, label="reference")
        else:
            self.content = self.fetch_content(
                recursive=self.recursive_search, label="reference"
            )

        # Fetch additional context if provided
        if context_files is not None:
            all_context = self.fetch_content(user_files=context_files, label="context")
            self.raw_context = "\n".join(all_context)

        # Create local embeddings
        self.encode_text_to_vector_store()

        # Check for testing mode
        if self.testing:
            self.web_search = False
            self.query = None
            self.raw_context = None
            self.polish = False
            self.context_size = "low"

    async def format_context(self, new_context, label="formatted_context"):
        """Creates more concise, less repetative context message."""

        # Format context references
        formatted_context = asyncio.run(
            self.single_completion(
                message=new_context, prompt_type="format_context"
            )
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
        if self.query is None:
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
        refined_query = refined_query["response"]
        result = self.save_response_text(refined_query, label="refined_query")
        self.logger.info(result)

        self.query = f"Ensure that addressing the following user query is the central theme of your response:\n{refined_query}\n\n"

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
            "formal_polishing" if self.tone == "default" else "modified_polishing"
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
        self.logger.info("Applying research agent to knowledge gaps...")

        # Identify gaps in understanding
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
            context = self.search_embedded_context(query=question, num_results=2)
            research_context += "\n" + context
        
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
