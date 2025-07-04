"""
This module provides functionality for evaluating the quality of generated summaries.

It includes a SummaryEvaluator class that uses various metrics to assess the faithfulness,
accuracy, and overall quality of summaries against their source content. The evaluation
process can be configured for testing purposes and supports multiple iterations for
consistent results.
"""

import asyncio

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase

from .openai import CompletionHandler
from .args import parse_eval_arguments


class SummaryEvaluator(CompletionHandler):
    """Attempt to objectively score the quality of a summary from a highly technical resource."""

    def __init__(
        self,
        summary: str,
        content: str,
        testing: bool = True,
        verbose: bool = True,
        iters: int = 5,
        **kwargs,
    ):
        """
        Initialize the SummaryEvaluator with the summary and content to evaluate.

        Args:
            summary: The generated summary to evaluate
            content: The original content that the summary is based on
            testing: Whether to run in testing mode
            verbose: Whether to print detailed output
            iterations: Number of iterations to run for evaluation
        """
        super().__init__(**kwargs)

        # Assign attributes
        self.summary = summary
        self.content = content
        self.iterations = iterations if iterations >= 3 else 3  # Ensure iterations
        self.temperature = 0.9
        self.random_seed = 42
        self.top_p = 1.0
        self.query = ""
        self.added_context = ""
        self.testing = testing
        self.verbose = verbose

        # Start async session
        self.initialize_session()

    def initialize_session(self):
        """Initialize a new session"""
        self._generate_run_tag()
        self._start_logging()
        self._create_output_path()
        self._read_system_instructions()
        self._new_openai_client()

    async def evaluate(self) -> Dict:
        """Evaluate summary text of a paired reference"""
        try:
            self.summary_evals = self.evaluate_summary()
        except Exception as specific_error:
            self.message = (
                f"An error occurred during summary evaluation: {str(specific_error)}"
            )
            print(self.message)
            raise RuntimeError("Summary evaluation failed") from specific_error

        self.evaluation = self.evaluate_summary()

        # Handle target text
        self.message = self.get_content_and_summary_text(self.content, self.summary)

        # Generate n responses to request
        await self._generate_eval_iterations()

        # Condense all evals to single response
        await self._condense_iterations()

        # Report final evaulation
        self.save_response_text(self.evaluation, label="evaluation")

        return self.evaluation

    def get_content_and_summary_text(self, content: str, summary: str) -> str:
        """Return combined content and summary text, reading from file if path provided."""

        # Determine content text
        try:
            content_text = self.fetch_content(user_files=[content], label="content")
        except Exception as e:
            content_text = content

        # Determine summary text
        try:
            summary_text = self.fetch_content(user_files=[summary], label="summary")
        except Exception as e:
            summary_text = summary

        # Form combined content and summary string
        message = f"""
        The following content is the original target text of expert summary generating agent:
        {content_text}

        Below here is the generated summary you are to score:
        {summary_text}
        """
        self.logger.info("Collected summary and target content text.")

        # Report new message
        self.save_response_text(message, label="eval_request")

        return message.strip()

    async def _generate_eval_iterations(self):
        """Generate multiple scoring responses for the provided summary text"""
        self.logger.info("Generating summary evaluation iterations...")

        # Asynchronously request iterative scoring text
        summary_evals = await self.multi_completions([self.message] * self.iterations)

        # Extract and save response strings
        self.summary_evals = [summary["response"] for summary in summary_evals]
        result = self.save_response_text(
            "\n".join(self.summary_evals), label="summary_eval_iters"
        )
        self.logger.info(result)

    async def _condense_iterations(self):
        """Condenses multiple API responses into a single coherent message."""
        self.logger.info("Condensing final evaluation text...")

        # Combine strings into demarcated single message
        combined_summaries = self._gen_iteration_str(self.summary_evals)

        # Obtain final score text
        condensed = await self.single_completion(
            message=combined_summaries,
            prompt_type="condense_scoring",
        )

        self.evaluation = condensed["response"]
        self.save_response_text(self.evaluation, label="final_eval")

    def _gen_iteration_str(self, evals):
        """Format single string with eval iteration text"""
        compiled_str = "\n\n".join(
            [
                "\n".join([f"Summary iteration: {i + 1}", evals[i].strip()])
                for i in range(len(evals))
            ]
        )

        return compiled_str.strip()


class SummaryScoring:
    """
    A class to evaluate the fidelity and overall quality of AI-generated summary text
    against an original article using the deepeval Python package.
    """

    def __init__(self):
        """
        Initializes the SummaryEvaluator.

        Note: For GEval to work, you must have an LLM provider configured
        (e.g., by setting environment variables like OPENAI_API_KEY).
        """
        print(
            "SummaryEvaluator initialized. Please ensure your LLM API key (e.g., OPENAI_API_KEY) "
            "is set as an environment variable if you plan to use GEval."
        )

    def _evaluate_faithfulness(
        self, original_article: str, generated_summary: str
    ) -> dict:
        """
        Evaluates the faithfulness of the generated summary to the original article.
        Faithfulness checks whether the generated summary contains factual information
        that is consistent with the original article.

        Args:
            original_article (str): The full text of the original article.
            generated_summary (str): The AI-generated summary text.

        Returns:
            dict: A dictionary containing the faithfulness score and explanation.
        """
        try:
            # Create a test case for Faithfulness evaluation
            # The input is the original article, the actual output is the generated summary.
            # The faithfulness metric checks if actual_output is supported by input.
            test_case = LLMTestCase(
                input=original_article,
                actual_output=generated_summary,
            )

            # Initialize the FaithfulnessMetric
            faithfulness_metric = FaithfulnessMetric(
                threshold=0.7
            )  # Adjust threshold as needed

            # Evaluate the test case with the metric
            faithfulness_metric.measure(test_case)

            return {
                "metric": "Faithfulness",
                "score": faithfulness_metric.score,
                "reason": faithfulness_metric.reason,
                "threshold_passed": faithfulness_metric.is_successful,
            }
        except Exception as e:
            return {
                "metric": "Faithfulness",
                "error": f"An error occurred during faithfulness evaluation: {e}",
            }

    def _evaluate_g_eval(self, original_article: str, generated_summary: str) -> dict:
        """
        Evaluates the overall quality of the generated summary using custom criteria
        via deepeval's GEval metric. This leverages an LLM to judge the summary.

        Criteria include: Coherence, Conciseness, Completeness, Readability, and Style.

        Args:
            original_article (str): The full text of the original article.
            generated_summary (str): The AI-generated summary text.

        Returns:
            dict: A dictionary containing the GEval score and explanation for each criterion.
        """
        try:
            # Define GEval criteria (evaluation steps)
            # These instructions guide the LLM on how to judge the summary quality.
            evaluation_steps = [
                "Assess if the summary is coherent and flows logically.",
                "Evaluate if the summary is concise and avoids unnecessary verbosity.",
                "Determine if the summary captures all the essential information from the original article.",
                "Judge the readability and clarity of the summary's language.",
                "Assess the overall grammar, spelling, and writing style of the summary.",
            ]

            # Create a GEval metric instance
            # The name helps identify the metric, and evaluation_params map to LLMTestCase fields.
            g_eval_metric = GEval(
                name="Summary Quality",
                evaluation_params=[
                    "input",
                    "actual_output",
                ],  # input is original, actual_output is summary
                criteria="Evaluate the summary's quality based on coherence, conciseness, completeness, readability, and style.",
                evaluation_steps=evaluation_steps,
                threshold=0.7,  # Overall threshold for GEval success
            )

            # Create a test case for GEval.
            # The 'input' here is the original context, 'actual_output' is what's being evaluated.
            test_case = LLMTestCase(
                input=original_article,
                actual_output=generated_summary,
            )

            # Measure the GEval score
            g_eval_metric.measure(test_case)

            # Extract detailed scores and explanations from GEval results
            results = {
                "metric": "GEval - Summary Quality",
                "overall_score": g_eval_metric.score,
                "overall_threshold_passed": g_eval_metric.is_successful,
                "criteria_details": {},
            }

            # GEval results provide individual scores for each evaluation step
            if (
                hasattr(g_eval_metric, "evaluation_results")
                and g_eval_metric.evaluation_results
            ):
                for res in g_eval_metric.evaluation_results:
                    results["criteria_details"][res.evaluation_step] = {
                        "score": res.score,
                        "reason": res.reason,
                    }
            else:
                results["reason"] = g_eval_metric.reason  # Fallback for overall reason

            return results

        except Exception as e:
            return {
                "metric": "GEval - Summary Quality",
                "error": f"An error occurred during GEval evaluation. Ensure your LLM environment variables are set correctly: {e}",
            }

    def evaluate_summary(self, original_article: str, generated_summary: str) -> dict:
        """
        Performs a comprehensive evaluation of the AI-generated summary using multiple
        deepeval metrics.

        Args:
            original_article (str): The full text of the original article.
            generated_summary (str): The AI-generated summary text.

        Returns:
            dict: A dictionary containing the results of all evaluations.
        """
        print("\n--- Starting Summary Evaluation ---")
        results = {}

        # Evaluate Faithfulness
        print("Evaluating Faithfulness...")
        faithfulness_result = self._evaluate_faithfulness(
            original_article, generated_summary
        )
        results["faithfulness"] = faithfulness_result
        print(f"Faithfulness Score: {faithfulness_result.get('score', 'N/A')}")
        print(f"Faithfulness Reason: {faithfulness_result.get('reason', 'N/A')}")

        # Evaluate overall quality using GEval
        print("\nEvaluating Overall Summary Quality (GEval)...")
        g_eval_result = self._evaluate_g_eval(original_article, generated_summary)
        results["g_eval"] = g_eval_result
        print(f"GEval Overall Score: {g_eval_result.get('overall_score', 'N/A')}")
        if "criteria_details" in g_eval_result:
            print("GEval Criteria Details:")
            for criterion, detail in g_eval_result["criteria_details"].items():
                print(
                    f"  - {criterion}: Score={detail.get('score', 'N/A')}, Reason='{detail.get('reason', 'N/A')}'"
                )
        else:
            print(
                f"GEval Reason: {g_eval_result.get('reason', g_eval_result.get('error', 'N/A'))}"
            )

        print("\n--- Evaluation Complete ---")
        return results


async def main():
    """Main entry point for the tldr command"""

    # Evaluate summary with custom method
    evaluator1 = SummaryEvaluator(**parse_eval_arguments())
    review = await evaluator1.evaluate()
    print("\n--- Summary Review ---", review, "\n")

    # Evaluate summary with deepeval
    # evaluator2 = SummaryScoring(**parse_eval_arguments())
    # scores = evaluator2.evaluate_summary(
    #    original_article=evaluator1.content, generated_summary=review
    # )
    # print("\n--- Summary Evaluation Scores ---", scores, "\n")


def cli_main():
    """Command-line entry point to run the tldr script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
