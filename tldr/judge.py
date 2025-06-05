import sys
import asyncio

from .single_completion import CompletionHandler
from .io import read_system_instructions, read_file_content


class SummaryEvaluator(CompletionHandler):
    """Attempt to objectively score the quality of a summary from a highly technical resource."""

    def __init__(
        self,
        model="gpt-4o-mini",
        output_directory=".",
        verbose=True,
        iters=5,
        temp=0.9,
        rng=42,
        top_p=1.0,
        api_key=None,
    ):

        # Assign attr
        self.model = model
        self.output_directory = output_directory
        self.verbose = verbose
        self.iterations = iters
        self.temperature = temp
        self.random_seed = rng
        self.top_p = top_p

        # Read in system instructions
        self.instructions = read_system_instructions()

    def evaluate(
        self, content=None, content_path=None, summary=None, summary_path=None
    ):
        """Evaluate summary text of a paired reference"""

        # Handle target text
        message = self._get_content_and_summary_text()

        # Generate n responses to request
        self._generate_eval_iterations(message)

        # Condense all evals to single response
        self.evaluation = self._condense_iterations()

        # Report final evaulation
        save_response_text(
            self.evaluation, label="evaluation", output_dir=self.output_directory
        )

    def _get_content_and_summary_text(self):
        """Handle grabbing user-provided text files"""

        # Check user arguments
        if self.content == None and self.content_path == None:
            print("No original content text or filepath was provided.")
            sys.exit(1)
        elif self.summary == None and self.summary_path == None:
            print("No summary text or filepath was provided.")
            sys.exit(1)

        # Read target content and summary files
        if self.content == None:
            try:
                self.content = read_file_content(self.content_path)
            except FileNotFoundError:
                print("Original content file was not found.")
                sys.exit(1)
        if self.summary == None:
            try:
                self.summary = read_file_content(self.summary_path)
            except FileNotFoundError:
                print("Summary file was not found.")
                sys.exit(1)

        # Combine target and summary texts
        return self._update_request()

    def _create_request(self):
        """Form combined content and summary string"""

        full_message = """
        The following content is the original target text of expert summary generating agent:
        {self.content}

        Below here is the generated summary you are to score:
        {self.summary}
        """

        return full_message.strip()

    async def _generate_eval_iterations(self, message):
        """Generate multiple scoring responses for the provided summary text"""

        # Ensure replication
        iters = int(self.iterations) if self.iterations >= 3 else 3

        # Request repeated scoring text
        if self.verbose is True:
            print("Generating summary evaluation iterations...")

        # Create coroutines for each iteration
        coroutines = [
            self.single_completion(
                message=self.summary,
                prompt_type="rubric_instructions",
                seed=self.random_seed + i,  # Add offset to seed for variation
                temperature=self.temperature,
                top_p=self.top_p,
            )
            for i in range(iters)
        ]

        # Run all coroutines concurrently and gather results
        self.evaluation_iters = await asyncio.gather(*coroutines)

    def _condense_iterations(self):
        """Condenses multiple API responses into a single coherent message."""
        if self.verbose is True:
            print("Condensing final evaluation text...")

        # Combine strings into demarcated single message
        responses = self._gen_iteration_str(self.evaluation_iters)

        # Obtain final score text
        condensed = asyncio.run(
            self.single_completion(
                message=responses,
                prompt_type="rubric_instructions",
                seed=self.random_seed,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        )

        return condensed

    def _gen_iteration_str(self, responses):
        """Format single string with response iteration text"""
        compiled_str = "\n\n".join(
            [
                "\n".join([f"Summary iteration: {i + 1}", responses[i].strip()])
                for i in range(len(responses))
            ]
        )

        return compiled_str
