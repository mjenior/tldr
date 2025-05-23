import os
import asyncio

from openai import OpenAI, AsyncOpenAI

from tldr.i_o import (
    create_timestamp,
    fetch_content,
    read_system_instructions,
    save_response_text,
)
from tldr.completion import CompletionHandler


class TldrClass(CompletionHandler):
    """
    Main TLDR class
    """

    def __init__(
        self,
        search_directory=".",
        output_directory=".",
        context_directory=None,
        verbose=True,
        api_key=None,
        token_scale="medium",
    ):
        # Set basic attr
        self.verbose = verbose
        self.token_scale = token_scale
        self.user_query = "\n"
        self.context = "\n"

        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

        # Read in files and contents
        if self.verbose == True:
            print("Searching for input files...")
        self.content = fetch_content(search_directory)
        self.sources = sum([len(self.content[x]) for x in self.content.keys()])

        # Fetch additional context if provided
        if context_directory is not None:
            raw_context = fetch_content(search_directory, combine=True)
            add_context = f", and {len(raw_context)} context documents"
            self.context = self.single_completion(
                message="\n".join(raw_context), prompt_type="format_context"
            )
        else:
            add_context = ""
        if self.verbose == True:
            print(f"Identified {self.sources} reference documents{add_context}.\n")

        # Read in system instructions
        self.instructions = read_system_instructions()

        # Establish output directory
        self.output_directory = self._handle_output_path(output_directory, verbose)

    def __call__(self, query=None):
        """
        More streamlined, callable interface to run the main TLDR pipeline.

        Arguments:
        - query: user query for focused summary context
        """

        async def _pipeline():

            # Update user query
            if query is not None:
                self.user_query = self.refine_user_query(query)

            # Generate async summaries
            self.all_summaries = await self.summarize_resources()

            # Integrate summaries
            if len(self.all_summaries) >= 2:
                self.integrate_summaries()
            else:
                self.content_synthesis = self.all_summaries[0]

            # Apply research if needed
            self.apply_research()

            # Polish the final result
            self.polish_response()

        # Run the async pipeline
        asyncio.run(_pipeline())

    @staticmethod
    def _handle_output_path(output_path, verbose=True) -> str:
        """Set up where to write output summaries"""

        # Create new path string
        current = create_timestamp()
        new_path = f"{output_path}/tldr.{current}"

        # Check if exists
        os.makedirs(new_path, exist_ok=True)

        if verbose == True:
            print("Output files being written to:", new_path)

        return new_path

    @staticmethod
    def _lint_user_query(current_query) -> None:
        """
        Normalizes `self.user_query` to ensure it's a well-formed string query.
        """

        # 1. Handle list input or convert to string
        if isinstance(current_query, list):
            processed_query = " ".join(map(str, current_query))
        elif isinstance(current_query, str):
            processed_query = current_query
        elif current_query is None:
            processed_query = ""
        else:
            processed_query = str(current_query)

        # Capitalize the first letter (if the string is not empty)
        if processed_query:  # Check if the string is not empty
            processed_query = processed_query[0].upper() + processed_query[1:]

        # Ensure the query ends with a question mark (if the string is not empty)
        if processed_query and not processed_query[-1] not in [".", "!", "?"]:
            processed_query += "?"
        elif not processed_query:
            pass

        return processed_query

    def refine_user_query(self, query):
        """Attempt to automatically improve user query for greater specificity"""

        if self.verbose == True:
            print("Refining user query...")

        # Check text formatting
        user_query = _lint_user_query(query)

        # Generate new query text
        new_query = self.single_completion(
            message=user_query, prompt_type="refine_prompt"
        )

        # Handle output text
        new_query = "Ensure that addressing the following user query is the key consideration in you response:\n{new_query}\n\n"
        save_response_text(
            new_query,
            label="new_query",
            output_dir=self.output_directory,
            verbose=self.verbose,
        )
        if self.verbose == True:
            print("Refined query:\n{new_query}\n")

        return new_query

    async def summarize_resources(self):
        """Generate component and synthesis summary text"""

        # Asynchronously summarize documents
        if self.verbose == True:
            print("Generating summaries for selected resources...")
        summaries = await self.async_multi_completions()
        await self.async_client.close()

        # Join response strings
        all_summaries = []
        for i in range(0, len(summaries)):
            all_summaries.append(f"Reference {i+1} Summary\n{summaries[i]}\n")
        save_response_text(
            all_summaries,
            label="summary",
            output_dir=self.output_directory,
            verbose=self.verbose,
        )

        return all_summaries

    def integrate_summaries(self, glyph_synthesis=False):
        """Generate integrated executive summaries combine all current summary text"""
        if self.verbose == True:
            print("Synthesizing integrated summary...")

        # Join all summaries for one large submission
        user_prompt = "\n\n".join(self.all_summaries)

        if glyph_synthesis == True:

            # Generate glyph-formatted prompt
            glyph_synthesis = self.single_completion(
                message=user_prompt, prompt_type="glyph_prompt"
            )
            save_response_text(
                glyph_synthesis,
                label="glyph_synthesis",
                output_dir=self.output_directory,
                verbose=self.verbose,
            )

            # Summarize documents based on glyph summary
            synthesis = self.single_completion(
                message=glyph_synthesis, prompt_type="interpret_glyphs"
            )
        else:
            # Otherwise, use standard synthesis system instructions prompt
            synthesis = self.single_completion(
                message=user_prompt, prompt_type="executive_summary"
            )

        # Handle output
        save_response_text(
            synthesis,
            label="synthesis",
            output_dir=self.output_directory,
            verbose=self.verbose,
        )
        self.content_synthesis = synthesis

    def apply_research(self, context_size="medium"):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""

        # Identify gaps in understanding
        if self.verbose == True:
            print("\rIdentifying technical gaps in summary...")
        gap_questions = self.single_completion(
            message=self.content_synthesis, prompt_type="research_instructions"
        )

        # Search web for to fill gaps
        if self.verbose == True:
            print("\rResearching gaps in important background...")
        research_results = self.search_web(gap_questions, context_size=context_size)

        # Handle output
        research_text = f"""Gap questions:
{gap_questions}

Answers:
{research_results}
"""
        save_response_text(
            research_text,
            label="research",
            output_dir=self.output_directory,
            verbose=self.verbose,
        )

        # Add to extra context
        self.context += research_results

    def polish_response(self, tone: str = "default"):
        """Refine final response text"""
        if self.verbose == True:
            print("Finalizing response text...")

        # Select tone
        tone_instructions = (
            "polishing_instructions" if tone == "default" else "modified_polishing"
        )
        response_text = self.user_query + self.content_synthesis

        # Run completion and save final response text
        polished = self.single_completion(
            message=response_text, prompt_type=tone_instructions
        )
        save_response_text(
            polished,
            label="final",
            output_dir=self.output_directory,
            verbose=self.verbose,
        )

        # Print final answer to terminal
        if self.verbose == True:
            print(f"\n\n{polished}\n")
