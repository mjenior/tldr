
# UNDER CONSTRUCTION

#chat = client.aio.chats.create(model='gemini-2.0-flash-001')
#response = await chat.send_message('tell me a story')

import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from openai import AsyncOpenAI

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

    def __init__(self, context_size="medium", api_key=None):
        super().__init__()
        self.context_size = context_size
        self.api_key = api_key

    def _new_openai_client(self):
        """Initialize OpenAI client"""
        # Set API key
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided.")

        # Initialize OpenAI clients
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.logger.info(f"Initialized OpenAI client: version={self.client._version}, context size={self.context_size}")

    def _scale_token(self, tokens):
        """Multiply the size of maximum output tokens allowed."""
        scaling = {"low": 0.5, "medium": 1.0, "high": 1.5}
        factor = scaling[self.context_size]
        return int(tokens * factor)


    def assemble_messages(self, message, prompt_type, web_search=True):
        """Assemble messages object, adding query and context information."""
        instructions_dict = self.instructions[prompt_type]
        instructions = instructions_dict["system_instruction"]
        output_tokens = self._scale_token(instructions_dict["max_output_tokens"])
        model = instructions_dict["model"]

        # Check message
        if len(message.strip()) == 0:
            raise ValueError("Message cannot be empty.")

        # Model-specific tools and reasoning
        if web_search is True:
            tools = [
                {"type": "web_search_preview", "search_context_size": self.context_size}
            ]
        else:
            tools = None
        if model == "o4-mini":
            reasoning = {"effort": self.context_size}
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
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.error.RateLimitError)
    )
    async def _perform_api_call(self, prompt, prompt_type, tries, search, **kwargs):
        """
        Helper function to encapsulate the actual API call logic.
        """
        call_params = self.assemble_messages(prompt, prompt_type, search)

        # Run completion query
        response = await self.client.responses.create(**call_params, **kwargs)

        # Update usage and return prompt and response pair
        self.update_usage(call_params["model"], response)
        return {"prompt": prompt, "response": response.output_text}

    def update_usage(self, model, response):
        """Update usage statistics."""
        self.model_tokens[model]["input"] += response.usage.input_tokens
        self.model_tokens[model]["output"] += response.usage.output_tokens
        self.update_spending()
        self.update_session_totals()
    
    async def _extract_sleep_time(self, error_message, adjust=0.5):
        """
        Extract the sleep time from the error message.

        Args:
            error_message: The error message from the OpenAI API.
            adjust: The amount of time to adjust the sleep time by.

        Returns:
            The sleep time in seconds.
        """
        try:
            sleep_time = float(error_message.split("Please try again in ")[1].split(".")[0])
            sleep_time += adjust
        except Exception as e:
            raise e

        return sleep_time

    async def single_completion(self, prompt, prompt_type, max_tries=9, web_search=True, **kwargs):
        """
        Initialize and submit a single chat completion request with built-in retry logic.
        """
        return await self._perform_api_call(
            prompt,
            prompt_type,
            tries=max_tries,
            search=web_search,
            **kwargs
        )

    async def multi_completions(self, message_list, max_tries=9, **kwargs):
        """
        Runs multiple single_completion calls concurrently using asyncio.gather, with retry on 429.
        """
        # Parse all of the content found
        coroutine_tasks = []
        for message in message_list:

            # Determine type of submission
            source_type = await self._perform_api_call(
                message, "file_type", max_tries=max_tries, **kwargs
            )

            # Generate summary task
            task = self._perform_api_call(
                message, source_type["response"], max_tries=max_tries, **kwargs
            )
            coroutine_tasks.append(task)

        return await asyncio.gather(*coroutine_tasks, return_exceptions=False)

