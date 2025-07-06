import os
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from google import genai
from google.genai import types


from .search import ResearchAgent

class CompletionHandlerGoogle(ResearchAgent):
    """
    Handles interactions with the Google Gemini completion API.

    This class is responsible for initializing the Google Gemini client, assembling
    messages for different prompt types, and managing the context size for
    API requests. It inherits from ResearchAgent to leverage its research
    capabilities.

    Attributes:
        context_size (str): The context window size for the model, affecting
            the number of output tokens. Can be "low", "medium", or "high".
        api_key (str, optional): The Google Gemini API key. If not provided, it will
            be read from the GEMINI_API_KEY environment variable.
        client (genai.Client): An asynchronous client for the Google Gemini API.
    """

    def __init__(self, context_size="medium", api_key=None, injection_screen=True):
        super().__init__()
        self.context_size = context_size
        self.api_key = api_key
        self.injection_screen = injection_screen

    def _new_gemini_client(self, version="v1beta"):
        """Initialize Gemini client"""
        # Set API key
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API key not provided.")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key, http_options={"api_version": version})
        self.logger.info(
            f"Initialized Google Gemini client; version={version}"
        )
        if self.injection_screen is True:
            self.logger.info("Prompt injection screen enabled.")
        else:
            self.logger.info("Prompt injection screen disabled.")

    def assemble_config(self, prompt_type, web_search=False, context_size="medium"):
        """Assemble configuration for API call."""
        role = self.prompt_dictionary[prompt_type.lower()]
        model = role["gemini_model"]
        
        # Scale dynamic threshold
        thinking_dict = {"low": 0, "medium": 0.4, "high": 0.8}
        thinking = types.ThinkingConfig(thinking_budget=thinking_dict[context_size])

        if web_search is True:
            search_dict = {"low": 0.25, "medium": 0.5, "high": 0.75}
            tools = [
                genai.types.Tool(
                    google_search=genai.types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=genai.types.DynamicRetrievalConfig(dynamic_threshold=search_dict[context_size])
                    )
                )
            ]
        else:
            tools = None

        # Build config object
        config = types.GenerateContentConfig(
            temperature=role["temperature"],
            system_instruction=role["system_instruction"],
            max_output_tokens=role["max_output_tokens"],
            tools=tools,
            thinking_config=thinking
        )

        return model, config


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
        model, config = self.assemble_config(prompt_type, web_search, context_size)

        # Run completion query with timeout
        try:
            # No retries, handle errors locally
            response = await asyncio.wait_for(
                self.client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=config,
                    **kwargs,
                ),
                timeout=timeout_seconds,
            )

            # Update usage and return prompt and response pair
            self.update_usage(model, response)
            return {
                "prompt": prompt,
                "response": response.text,
                "source": source,
            }
        except asyncio.TimeoutError:
            self.logger.error("API call timed out.")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in API call: {str(e)}")
            raise

    def update_usage(self, model, response):
        """Update usage statistics."""
        usage_metadata = response.usage_metadata
        self.model_tokens[model]["input"] += usage_metadata.input_token_count
        self.model_tokens[model]["output"] += usage_metadata.output_token_count
        self.update_spending()
        self.update_session_totals()

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