import os
import sys
import math
import logging
from typing import List, Optional

from .io import FileHandler


class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that updates Streamlit UI with log messages."""

    def __init__(self):
        super().__init__()
        self.logs: List[str] = []
        self.max_logs = 1000  # Maximum number of logs to keep in memory

    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)
            self.logs.append(msg)
            # Keep only the most recent logs
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs :]
        except Exception:
            self.handleError(record)

    def get_logs(self) -> List[str]:
        """Get the current list of log messages."""
        return self.logs

    def clear_logs(self):
        """Clear all stored log messages."""
        self.logs = []


class StreamlitStdout:
    def __init__(self, streamlit_handler):
        self.streamlit_handler = streamlit_handler
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                if line.strip():
                    # Create a log record and emit it through the handler
                    record = logging.LogRecord(
                        "stdout", logging.INFO, "", 0, line, (), None
                    )
                    self.streamlit_handler.emit(record)
            self.buffer = lines[-1]

    def flush(self):
        if self.buffer:
            record = logging.LogRecord(
                "stdout", logging.INFO, "", 0, self.buffer, (), None
            )
            self.streamlit_handler.emit(record)
        self.buffer = ""


class LogHandler(FileHandler):
    """
    Manages logging and tracks API usage costs for the application.

    This class is responsible for two main things:
    1.  Configuring the application's logger to provide detailed output.
    2.  Tracking the number of tokens used and the associated costs for
        different OpenAI models during a session.

    It inherits from `FileHandler` to leverage file I/O capabilities.

    Attributes:
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

    def __init__(self, platform="openai", verbose=True):
        super().__init__()
        self.verbose = verbose
        self.platform = platform
        self.streamlit_handler: Optional[StreamlitLogHandler] = None
        self.logger = logging.getLogger()

        if self.platform == "openai":
            low_model = "gpt-4o-mini"
            medium_model = "gpt-4o"
            high_model = "o4-mini"
            self.model_rates_perM = {
                low_model: {"input": 0.15, "output": 0.6},
                medium_model: {"input": 2.5, "output": 10.0},
                high_model: {"input": 1.1, "output": 4.4},
            }
        elif self.platform == "gemini":
            low_model = "gemini-2.0-flash"
            medium_model = "gemini-2.5-flash"
            high_model = "gemini-2.5-pro"
            self.model_rates_perM = {
                low_model: {"input": 0.1, "output": 0.4},
                medium_model: {"input": 0.3, "output": 2.5},
                high_model: {"input": 1.25, "output": 10.0},
            }

        self.model = low_model
        self.model_tokens = {
            low_model: {"input": 0, "output": 0},
            medium_model: {"input": 0, "output": 0},
            high_model: {"input": 0, "output": 0},
        }
        self.model_spend = {
            low_model: {"input": 0.0, "output": 0.0},
            medium_model: {"input": 0.0, "output": 0.0},
            high_model: {"input": 0.0, "output": 0.0},
        }
        self.session_spend = {"input": 0.0, "output": 0.0}
        self.total_spend = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _start_logging(self):
        """
        Configures the root logger for the application.
        """
        # Reset logger to ensure clean state
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Always create StreamlitLogHandler for potential Streamlit use
        self.streamlit_handler = StreamlitLogHandler()
        self.streamlit_handler.setFormatter(formatter)
        self.logger.addHandler(self.streamlit_handler)

        # Always add console handler for stdout logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(self.output_directory, f"{self.run_tag}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info("Logging system initialized")
        self.logger.info(
            f"Intermediate files being written to: {self.output_directory}"
        )

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
