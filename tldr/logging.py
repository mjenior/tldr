import os
import sys
import math
import logging

from .i_o import create_timestamp


class ExpenseTracker:
    """
    Tracker for token useage and session cost
    """

    def __init__(self):
        self.model = "gpt-4o-mini"

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

    @staticmethod
    def _simplify_usd_str(usd: float, dec: int = 2) -> str:
        """Returns a slightly more human-readable USD string, always rounding UP."""
        factor = 10**dec
        rounded = math.ceil(usd * factor) / factor
        return f"${rounded:.{dec}f}"


def setup_logging(verbose: bool):
    """
    Configures the root logger for the application.
    """
    # Get root logger
    logger = logging.getLogger()

    # Prevent duplicate handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Set the verbosity
    (
        logger.setLevel(logging.INFO)
        if verbose == True
        else logger.setLevel(logging.WARNING)
    )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # File handler
    run_tag = f"tldr.{create_timestamp()}"
    log_file = os.path.join(f"{run_tag}.log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return run_tag


def get_logfile_path(logger_instance: logging.Logger = None) -> str | None:
    """
    Retrieves the filename of the first FileHandler attached to the given logger.
    """
    if logger_instance is None:
        logger_instance = logging.getLogger()

    for handler in logger_instance.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None
