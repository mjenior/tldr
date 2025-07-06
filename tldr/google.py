import os
import math
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

class CompletionHandler(ResearchAgent):
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

    def _new_openai_client(self):
        """Initialize OpenAI client"""
        # Set API key
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API key not provided.")

        # Initialize OpenAI clients
        self.client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
        self.logger.info(
            f"Initialized OpenAI client: version={self.client._version}, context size={self.context_size}"
        )
        if self.injection_screen is True:
            self.logger.info("Prompt injection screen enabled.")
        else:
            self.logger.info("Prompt injection screen disabled.")

    def assemble_config(self, prompt_type, web_search=False):
        """Assemble configuration for API call."""
        role = self.prompt_dictionary[prompt_type.lower()]
        model = role["gemini_model"]
        
        

        # TODO: Scale output tokens

        if web_search is True:
            tools = [
                genai.types.Tool(
                    google_search=genai.types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=genai.types.DynamicRetrievalConfig(dynamic_threshold=0.7)
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
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
        )
        config = {
                "": role["temperature"],temperature
                "system_instruction": role["system_instruction"],
                "max_output_tokens": role["max_output_tokens"],
                "tools": tools,
                "reasoning": reasoning,
            }
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
        except asyncio.TimeoutError:
            self.logger.error("API call timed out.")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in API call: {str(e)}")
            raise


client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Write a story about a magic backpack."
)
print(response.text)
    
    