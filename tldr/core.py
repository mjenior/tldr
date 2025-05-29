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
        search_directory=".",
        output_directory=".",
        context_directory=None,
        recursive_search=False,
        verbose=True,
        api_key=None,
        context_size="medium",
        testing=False,
    ):
        super().__init__()

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

        # Establish output directory
        self.output_directory = self._handle_output_path(output_directory)

        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Read in files and contents
        self.logger.info("Searching for input files...")
        self.content = fetch_content(search_directory, recursive=self.recursive_search)
        self.sources = sum([len(self.content[x]) for x in self.content.keys()])
        if self.verbose == True:
            print(f"Identified {self.sources} reference documents for summarization.")

        # Fetch additional context if provided
        if context_directory is not None:
            all_context = fetch_content(
                context_directory, combine=True, recursive=self.recursive_search
            )
            if len(all_context) >= 1 and self.verbose == True:
                print(
                    f"Also identified {self.sources} reference documents for added context."
                )

            self.raw_context = "\n".join(all_context)
        else:
            self.raw_context = None

        self.logger.info(f"Output files being written to: {self.output_directory}")

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

    @staticmethod
    def _handle_output_path(output_path) -> str:
        """Set up where to write output summaries"""

        # Create full path for intermediate files
        temp_path = os.path.join(
            output_path, f"tldr.{create_timestamp()}", "intermediate"
        )
        os.makedirs(temp_path, exist_ok=True)

        # Return top level new directory
        new_path = os.path.join(output_path, f"tldr.{create_timestamp()}")

        return new_path

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

        # Annotate response strings
        all_summaries = [
            f"Reference {i+1} Summary\n{summaries[i]}\n"
            for i in range(0, len(summaries))
        ]
        result = save_response_text(
            all_summaries,
            label="summary",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.all_summaries = all_summaries

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
        result = save_response_text(
            f"Gap questions:\n{gap_questions}\n\nAnswers:\n{research_results}",
            label="research",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.added_context += research_results

    async def polish_response(self, tone: str = "default"):
        """Refine final response text"""

        # Select tone
        tone_instructions = (
            "polishing_instructions" if tone == "default" else "modified_polishing"
        )

        # Run completion and save final response text
        polished_text = await self.single_completion(
            message=self.content_synthesis, prompt_type=tone_instructions
        )
        polished_title = await self.single_completion(
            message=polished_text, prompt_type="title_instructions"
        )

        # Save formatted PDF
        pdf_path = generate_tldr_pdf(
            polished_text, polished_title, self.output_directory
        )
        self.polished_summary = polished_text

        return f"Final summary saved to {pdf_path}\n"
