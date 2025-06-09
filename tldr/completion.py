import os
import asyncio
from functools import partial

from openai import AsyncOpenAI

from .logger import ExpenseTracker


class CompletionHandler(ExpenseTracker):

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

        # Assemble messages object, adding query and context information
        messages = [
            {"role": "system", "content": instructions + self.user_query},
            {"role": "user", "content": message + self.added_context},
        ]

        # Rescale max tokens
        max_output_tokens = self._scale_token(output_tokens)

        # Run completion query
        if model == "o4-mini":
            response = await self.client.responses.create(
                input=messages,
                model=model,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": self.context_size},
                **kwargs,
            )
        else:
            response = await self.client.responses.create(
                input=messages,
                model=model,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )

        # Update usage
        self.model_tokens[model]["input"] += response.usage.input_tokens
        self.model_tokens[model]["output"] += response.usage.output_tokens
        self.update_spending()
        self.update_session_totals()

        # Return only the response text
        return response.output_text

    async def single_completion(self, message, prompt_type, **kwargs):
        """
        Initialize and submit a single chat completion request with built-in retry logic.
        """
        return await self._retry_on_429(
            partial(self._perform_api_call, **kwargs),
            message=message,
            prompt_type=prompt_type,
        )

    async def multi_completions(self, **kwargs):
        """
        Runs multiple single_completion calls concurrently using asyncio.gather, with retry on 429.
        """
        # Parse all of the content found
        coroutine_tasks = []
        for content in self.content:

            # Determine type of resource
            source_type = await self._retry_on_429(
                partial(self._perform_api_call, **kwargs),
                message=content,
                prompt_type="file_type",
            )

            # Generate summary
            task = self._retry_on_429(
                self.single_completion,
                message=content,
                prompt_type=source_type,
            )
            coroutine_tasks.append(task)

        return await asyncio.gather(*coroutine_tasks, return_exceptions=False)

    async def _retry_on_429(self, coro_fn, message, prompt_type, max_retries=5):
        """Retry summary generation on token rate limit per hour exceeded errors."""

        for attempt in range(1, max_retries + 1):
            try:
                return await coro_fn(message, prompt_type)
            except Exception as e:
                if hasattr(e, "status") and e.status == 429:
                    wait_time = random.uniform(*(1, 3)) * attempt * 2
                    self.logger.info(
                        f"Rate limit hit. Retrying in {wait_time:.2f}s (attempt {attempt})..."
                    )
                    await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded for task")

    async def _search_single_question(self, question: str, context_size: str):
        """Helper function to search for a single question using web_search_preview tool."""


        # TODO: use instructions here to consolidate api cal function
        instruction = "Use web search to answer the following question as thoroughly as possible. Cite all sources.\n"
        response = await self.client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview", "search_context_size": context_size}],
            input=instruction + question,
        )
        return response.output_text

    async def search_web(self, questions: list[str], context_size: str = "medium"):
        """
        Use web search tool to answer a list of questions asynchronously.
        Each question is processed individually.

        Args:
            questions (list[str]): A list of questions to search the web for.
            context_size (str, optional): The context size for the web search. 
                                         Defaults to "medium".

        Returns:
            list[tuple[str, str | None]]: A list of tuples, where each tuple contains 
                                          the original question and its corresponding answer text.
                                          If an error occurs for a question, the answer text will be None.
        """
        if not questions:
            return []

        tasks = [
            self._search_single_question(question, context_size)
            for question in questions
        ]
        
        # Gather results, allowing exceptions to be returned
        answers_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, item in enumerate(answers_or_exceptions):
            original_question = questions[i]
            if isinstance(item, Exception):
                error_message = f"Error searching web for question '{original_question}': {item}"
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.error(error_message)
            else:
                results.append((original_question, item)) # item is the answer_text
        
        return results
