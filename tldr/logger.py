import os
import sys
import math
import logging

from .io import FileHandler


class LogHandler(FileHandler):
    """
    Manages logging and tracks API usage costs for the application.

    This class is responsible for two main things:
    1.  Configuring the application's logger to provide detailed output.
    2.  Tracking the number of tokens used and the associated costs for
        different OpenAI models during a session.

    It inherits from `FileHandler` to leverage file I/O capabilities.

    Attributes:
        model (str): The default model for which to track usage.
        verbose (bool): A flag to control the verbosity of logging.
        model_rates_perM (dict): A dictionary mapping model names to their
            cost per million tokens for input and output.
        model_tokens (dict): A dictionary tracking the number of input and
            output tokens used per model.
        model_spend (dict): A dictionary tracking the cost of input and
            output tokens used per model.
        session_spend (dict): A dictionary tracking the total input and
            output cost for the current session.
        total_spend (float): The total cost for the current session.
        logger (logging.Logger): The logger instance for the application.
    """

    def __init__(self):
        super().__init__()
        self.model = "gpt-4o-mini"
        self.verbose = True

        # Define spending dictionaries
        self.model_rates_perM = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "o4-mini": {"input": 1.1, "output": 4.4},
        }
        self.model_tokens = {
            "gpt-4o-mini": {"input": 0, "output": 0},
            "gpt-4o": {"input": 0, "output": 0},
            "o4-mini": {"input": 0, "output": 0},
        }
        self.model_spend = {
            "gpt-4o-mini": {"input": 0.0, "output": 0.0},
            "gpt-4o": {"input": 0.0, "output": 0.0},
            "o4-mini": {"input": 0.0, "output": 0.0},
        }
        self.session_spend = {"input": 0.0, "output": 0.0}
        self.total_spend = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _start_logging(self):
        """
        Configures the root logger for the application.
        """
        # Get root logger
        self.logger = logging.getLogger()

        # Prevent duplicate handlers
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        # Set the verbosity
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # File handler
        self.logger.info(
            f"Intermediate files being written to: {self.output_directory}"
        )
        log_file = os.path.join(self.output_directory, f"{self.run_tag}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def update_spending(self):
        """Calculates approximate cost (USD) of LLM tokens generated to a given decimal place"""

        # Fetch correct model dictionary
        rate_dict = self.model_rates_perM.get(self.model)
        useage_dict = self.model_tokens.get(self.model)

        # Update record keeping dictionaries
        for direction in ["input", "output"]:
            self.model_spend[self.model][direction] += (
                useage_dict[direction] * rate_dict[direction]
            ) / 1e6
            self.total_input_tokens += useage_dict["input"]
            self.total_output_tokens += useage_dict["output"]

    def update_session_totals(self):
        """Calculate current seesion totals for token use and spending"""

        self.session_spend = {"input": 0.0, "output": 0.0}

        for model in self.model_spend.keys():
            self.session_spend["input"] += self.model_spend[model]["input"]
            self.session_spend["output"] += self.model_spend[model]["output"]

        self.total_spend = self.session_spend["input"] + self.session_spend["output"]

    def format_spending(self):

        self.session_spend["input"] = self._simplify_usd_str(
            self.session_spend["input"]
        )
        self.session_spend["output"] = self._simplify_usd_str(
            self.session_spend["output"]
        )
        self.total_spend = self._simplify_usd_str(self.total_spend)

        for model in self.model_spend.keys():
            self.model_spend[model]["input"] = self._simplify_usd_str(
                self.model_spend[model]["input"]
            )
            self.model_spend[model]["output"] = self._simplify_usd_str(
                self.model_spend[model]["output"]
            )

        self.logger.info(f"Session Cost: {self.total_spend}")

    def generate_token_report(self) -> str:
        """
        Generates a report string showing input/output tokens per model.
        """
        report = []
        for model, tokens in self.model_tokens.items():
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            report.append(f"{model} i:{input_tokens}, o:{output_tokens}")

        self.token_report = ", ".join(report)
        self.logger.info(f"Session Token Usage: {self.token_report}")

    @staticmethod
    def _simplify_usd_str(usd: float, dec: int = 2) -> str:
        """Returns a slightly more human-readable USD string, always rounding UP."""
        factor = 10**dec
        rounded = math.ceil(usd * factor) / factor
        return f"${rounded:.{dec}f}"
