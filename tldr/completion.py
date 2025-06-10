import os
import asyncio
from functools import partial

from openai import AsyncOpenAI

from .dartboard import DartboardRetriever


class CompletionHandler(DartboardRetriever):

    def __init__(self, testing=False, context_size="medium", api_key=None):
        super().__init__()

        self.testing = testing
        self.context_size = context_size

        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
        self.client = AsyncOpenAI(api_key=self.api_key)

    def _scale_token(self, tokens):
        """Multiply the size of maximum output tokens allowed."""
        scaling = {"low": 0.5, "medium": 1.0, "high": 1.5}
        factor = scaling[self.context_size]
        return int(tokens * factor)

    async def _perform_api_call(self, message, prompt_type, **kwargs):
        """
        Helper function to encapsulate the actual API call logic.
        """
        instructions_dict = self.instructions[prompt_type]
        instructions = instructions_dict["system_instruction"]
        output_tokens = instructions_dict["max_output_tokens"]
        model = instructions_dict["model"] if self.testing == False else "gpt-4o-mini"

        # Check message
        if len(message.strip()) == 0:
            raise ValueError("Message cannot be empty.")

        # Set tools
        if model == "gpt-4o":
            tools = [
                {"type": "web_search_preview", "search_context_size": self.context_size}
            ]
        else:
            tools = None

        # Assemble messages object, adding query and context information
        messages = [
            {"role": "system", "content": instructions + self.query},
            {"role": "user", "content": message + self.added_context},
        ]

        # Run completion query
        if model == "o4-mini":
            response = await self.client.responses.create(
                input=messages,
                model=model,
                max_output_tokens=self._scale_token(output_tokens),
                reasoning={"effort": self.context_size},
                **kwargs,
            )
        else:
            response = await self.client.responses.create(
                input=messages,
                model=model,
                tools=tools,
                max_output_tokens=self._scale_token(output_tokens),
                **kwargs,
            )

        # Update usage
        self.model_tokens[model]["input"] += response.usage.input_tokens
        self.model_tokens[model]["output"] += response.usage.output_tokens
        self.update_spending()
        self.update_session_totals()

        # Return prompt and response pair
        return {"prompt": message, "response": response.output_text}

    async def single_completion(self, message, prompt_type, **kwargs):
        """
        Initialize and submit a single chat completion request with built-in retry logic.
        """
        return await self._retry_on_429(
            partial(self._perform_api_call, **kwargs),
            message=message,
            prompt_type=prompt_type,
        )

    async def multi_completions(self, message_list, prompt_type="default", **kwargs):
        """
        Runs multiple single_completion calls concurrently using asyncio.gather, with retry on 429.
        """
        # Parse all of the content found
        coroutine_tasks = []
        for message in message_list:

            # Determine type of submission
            if prompt_type != "web_search":
                # Determine type of resource
                source_type = await self._retry_on_429(
                    partial(self._perform_api_call, **kwargs),
                    message=message,
                    prompt_type="file_type",
                )
                prompt_type = source_type["response"]

            # Generate summary task
            task = self._retry_on_429(
                partial(self._perform_api_call, **kwargs),
                message=message,
                prompt_type=prompt_type,
            )
            coroutine_tasks.append(task)

        return await asyncio.gather(*coroutine_tasks, return_exceptions=False)

    async def _retry_on_429(
        self, coro_fn, message, prompt_type, max_retries=9, **kwargs
    ):
        """Retry summary generation on token rate limit per hour exceeded errors."""

        for attempt in range(1, max_retries + 1):
            try:
                return await coro_fn(message, prompt_type, **kwargs)
            except Exception as e:
                if hasattr(e, "status") and e.status == 429:
                    wait_time = random.uniform(*(1, 3)) * attempt * 5
                    self.logger.info(
                        f"Rate limit hit. Retrying in {wait_time:.2f}s (attempt {attempt})..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise e

        raise RuntimeError(f"Max retries ({max_retries}) exceeded for task.")
