import os
import re
import logging

import asyncio
from openai import AsyncOpenAI

from .i_o import *
from .completion import CompletionHandler


class TldrClass(CompletionHandler):
    """
    Main TLDR class
    """

    def __init__(
        self,
        input_files=None,
        context_files=None,
        recursive_search=False,
        verbose=True,
        api_key=None,
        context_size="medium",
        tag="tldr_run",
        testing=False,
    ):
        super().__init__()

        # Establish output directory
        self._create_output_path(tag)

        # Pull in logger
        self.logger = logging.getLogger(__name__)

        # Set basic attr
        self.verbose = verbose
        self.testing = testing
        self.context_size = context_size
        self.recursive_search = recursive_search
        self.user_query = ""
        self.added_context = ""

        # Read in system instructions
        self.instructions = read_system_instructions()

        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Read in files and contents
        self.logger.info("Searching for input files...")
        if input_files is not None:
            self.content = fetch_content(user_files=input_files)
        else:
            self.content = fetch_content(recursive=self.recursive_search)
        self.logger.info(
            f"Identified {len(self.content)} reference documents for summarization."
        )

        # Fetch additional context if provided
        if context_files is not None:
            all_context = fetch_content(user_files=context_files)
            self.logger.info(
                f"Also identified {len(all_context)} reference documents for added context."
            )
            self.raw_context = "\n".join(all_context)
        else:
            self.raw_context = None

        self.logger.info(
            f"Intermediate files being written to: {self.output_directory}"
        )

    async def format_context_references(self):
        """Creates more concise, less repetative context message."""

        added_context = await self.single_completion(
            message="\n".join(self.raw_context), prompt_type="format_context"
        )
        result = save_response_text(
            added_context,
            label="added_context",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.added_context += f"\nBelow is additional context for reference during response generation:\n{added_context}"

    def _create_output_path(self, run_tag=None):
        """Set up where to write output summaries"""

        # Create full path for intermediate files
        if run_tag is not None:
            output_path = f"{run_tag}_files"
        else:
            output_path = f"tldr.{create_timestamp()}_files"
        os.makedirs(output_path, exist_ok=True)
        self.output_directory = output_path

    @staticmethod
    def _lint_user_query(current_query) -> str:
        """
        Normalizes a user query to ensure it's a well-formed, AI-friendly string.
        """

        # Normalize input to string
        if isinstance(current_query, list):
            processed_query = " ".join(map(str, current_query))
        elif current_query is None:
            processed_query = ""
        else:
            processed_query = str(current_query)

        # Strip leading/trailing whitespace and normalize internal whitespace
        processed_query = re.sub(r"\s+", " ", processed_query.strip())

        if processed_query:
            # Capitalize first letter if not already
            processed_query = processed_query[0].upper() + processed_query[1:]

            # Ensure it ends with a suitable punctuation mark
            if processed_query[-1] not in ".!?":
                processed_query += "?"

        return processed_query

    async def refine_user_query(self, query):
        """Attempt to automatically improve user query for greater specificity"""

        # Check text formatting
        user_query = self._lint_user_query(query)

        # Generate new query text
        new_query = await self.single_completion(
            message=user_query, prompt_type="refine_prompt"
        )

        # Handle output text
        full_query = f"Ensure that addressing the following user query is the key consideration in you response:\n{new_query}\n"
        result = save_response_text(
            new_query,
            label="new_query",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.user_query = full_query

    async def summarize_resources(self):
        """Generate component and synthesis summary text"""

        # Asynchronously summarize documents
        summaries = await self.multi_completions()

        # Annotate and save response strings
        self.all_summaries = ""
        i = 0
        for summary in summaries:
            i += 1
            annotated = f"Reference {i} Summary\n{summary}\n"
            result = save_response_text(
                annotated,
                label="summary",
                output_dir=self.output_directory,
                idx=i,
            )
            self.logger.info(result)
            self.all_summaries += annotated

    async def integrate_summaries(self):
        """Generate integrated executive summaries combine all current summary text"""

        # Join all summaries for one large submission
        summaries = "\n\n".join(self.all_summaries)
        synthesis = await self.single_completion(
            message=summaries, prompt_type="executive_summary"
        )

        # Handle output
        result = save_response_text(
            synthesis,
            label="synthesis",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.content_synthesis = synthesis

    async def apply_research(self):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""

        # Identify gaps in understanding
        gap_questions = await self.single_completion(
            message=self.content_synthesis, prompt_type="research_instructions"
        )

        # Search web for to fill gaps
        research_results = await self.search_web(
            gap_questions, context_size=self.context_size
        )

        # Handle output
        gap_gilling = f"Gap questions:\n{gap_questions}\n\nAnswers:\n{research_results}"
        result = save_response_text(
            gap_gilling,
            label="research",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.research_results = gap_gilling

    async def polish_response(self, tone: str = "default"):
        """Refine final response text"""

        # Select tone
        tone_instructions = (
            "polishing_instructions" if tone == "default" else "modified_polishing"
        )

        # Run completion and save final response text
        polished_text = await self.single_completion(
            message=self.content_synthesis + self.research_results,
            prompt_type=tone_instructions,
        )
        polished_title = await self.single_completion(
            message=polished_text, prompt_type="title_instructions"
        )

        # Save formatted PDF
        pdf_path = generate_tldr_pdf(polished_text, polished_title)
        self.polished_summary = polished_text

        return f"Final summary saved to {pdf_path}\n"
