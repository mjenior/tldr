import os
import math
import asyncio
from asyncio import TimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from openai import AsyncOpenAI, RateLimitError

from .search import ResearchAgent


class CompletionHandler(ResearchAgent):
    """
    Handles interactions with the OpenAI completion API.

    This class is responsible for initializing the OpenAI client, assembling
    messages for different prompt types, and managing the context size for
    API requests. It inherits from ResearchAgent to leverage its research
    capabilities.

    Attributes:
        context_size (str): The context window size for the model, affecting
            the number of output tokens. Can be "low", "medium", or "high".
        api_key (str, optional): The OpenAI API key. If not provided, it will
            be read from the OPENAI_API_KEY environment variable.
        client (AsyncOpenAI): An asynchronous client for the OpenAI API.
    """

    def __init__(self, context_size="medium", api_key=None, injection_screen=True):
        super().__init__()
        self.context_size = context_size
        self.api_key = api_key
        self.injection_screen = injection_screen

    def _new_openai_client(self):
        """Initialize OpenAI client"""
        # Set API key
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.logger.info(
            f"Initialized OpenAI client: version={self.client._version}, context size={self.context_size}"
        )
        if self.injection_screen is True:
            self.logger.info("Prompt injection screen enabled.")
        else:
            self.logger.info("Prompt injection screen disabled.")

    def _scale_token(self, tokens):
        """Multiply the size of maximum output tokens allowed."""
        scaling = {"low": 0.5, "medium": 1.0, "high": 1.5}
        factor = scaling[self.context_size]
        return int(tokens * factor)

    def assemble_messages(
        self, message, prompt_type, web_search=True, context_size="medium"
    ):
        """Assemble messages object, adding query and context information."""
        instructions_dict = self.prompt_dictionary[prompt_type.lower()]
        instructions = instructions_dict["system_instruction"]
        output_tokens = self._scale_token(instructions_dict["max_output_tokens"])
        model = instructions_dict["model"]

        # Check message
        if len(message.strip()) == 0:
            raise ValueError("Message cannot be empty.")

        # Model-specific tools and reasoning
        if web_search is True:
            tools = [
                {"type": "web_search_preview", "search_context_size": context_size}
            ]
        else:
            tools = None
        if model == "o4-mini":
            reasoning = {"effort": context_size}
        else:
            reasoning = None

        # Update query if needed
        if self.query is not None and self.query.strip() != "":
            query = f"Ensure that addressing the following user query is a central theme of your response:\n{self.query}\n\n"
        else:
            query = ""

        # Update added context if needed
        if self.added_context is not None and self.added_context.strip() != "":
            added_context = f"\n\nHere is additional context to consider when generating your response:\n{self.added_context}\n\n"
        else:
            added_context = ""

        # Assemble messages object, adding query and context information
        messages = [
            {"role": "system", "content": query + instructions},
            {"role": "user", "content": message + added_context},
        ]

        # Create parameter dictionary
        params = {
            "input": messages,
            "model": model,
            "max_output_tokens": output_tokens,
            "tools": tools,
            "reasoning": reasoning,
        }

        return params

    @retry(
        wait=wait_random_exponential(multiplier=2, min=5, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, asyncio.TimeoutError)),
    )
    async def perform_api_call(
        self,
        prompt,
        prompt_type,
        web_search=False,
        context_size="medium",
        source=None,
        timeout_seconds=30,
        injection_screen=True,
        **kwargs,
    ):
        """
        Helper function to encapsulate the actual API call logic.

        Args:
            prompt: The input prompt to send to the API
            prompt_type: Type of prompt (determines model and instructions)
            web_search: Whether to enable web search
            context_size: Size of the context window
            source: Source of the content (for tracking)
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dict containing the prompt, response, and source

        Raises:
            asyncio.TimeoutError: If the API call times out
            Exception: For other API errors
        """
        # Screen for prompt injection
        if injection_screen is True:
            self.logger.info(f"Testing for prompt injection...")
            injection_test = await self.detect_prompt_injection(prompt)
            if injection_test is True:
                self.logger.warning(f"Prompt injection detected: {prompt}")
                return {
                    "prompt": prompt,
                    "response": "Prompt injection detected. Skipped.",
                    "source": source,
                }
            else:
                self.logger.info("No prompt injection detected. Proceeding.")

        # Assemble messages and call parameters
        call_params = self.assemble_messages(
            prompt, prompt_type, web_search, context_size
        )

        # Run completion query with timeout
        try:
            # No retries, handle errors locally
            response = await asyncio.wait_for(
                self.client.responses.create(**call_params, **kwargs),
                timeout=timeout_seconds,
            )

            # Update usage and return prompt and response pair
            self.update_usage(call_params["model"], response)
            return {
                "prompt": prompt,
                "response": response.output_text,
                "source": source,
            }

        except TimeoutError:
            self.logger.error(f"API call timed out. Retrying...")
            await asyncio.sleep(1)
            raise

        except RateLimitError as e:
            self.logger.error(f"Current OpenAI API rate limit exceeded.")
            await self._ratelimit_sleep(str(e))
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error in API call: {str(e)}")
            raise

    def update_usage(self, model, response):
        """Update usage statistics."""
        self.model_tokens[model]["input"] += response.usage.input_tokens
        self.model_tokens[model]["output"] += response.usage.output_tokens
        self.update_spending()
        self.update_session_totals()

    async def _ratelimit_sleep(
        self, error_message: str, adjust: float = 0.25, dec: int = 2
    ) -> float:
        """
        Extract the sleep time from the error message.

        Args:
            error_message: The error message from the OpenAI API.
            adjust: The amount of time to adjust the sleep time by.

        Returns:
            The sleep time in seconds.
        """
        try:
            sleep_time = float(
                error_message.split("Please try again in ")[1].split(".")[0]
            )
        except Exception as e:
            if "429 Too Many Requests" in e:
                sleep_time = 5.0
            else:
                self.logger.error(f"Unexpected error in API call: {str(e)}")
                raise e

        sleep_time = self.round_up_to_decimal_place(sleep_time + adjust, dec)
        self.logger.warning(f"Waiting for {sleep_time} seconds before retrying")
        await asyncio.sleep(sleep_time)

    @staticmethod
    def round_up_to_decimal_place(num: float, dec_places: int = 2) -> float:
        """Rounds a number up to a specified number of decimal places."""
        places = 10**dec_places
        return math.ceil(num * places) / places

    async def detect_prompt_injection(self, prompt: str) -> bool:
        """
        Detect potential prompt injection and return boolean test result
        """
        test_result = await self.perform_api_call(
            prompt=prompt.strip(),
            prompt_type="detect_injection",
            context_size="low",
            injection_screen=False,  # Prevent infinite recursion
        )
        # Check test result
        if "yes" in test_result["response"].lower():
            return True
        else:
            return False
