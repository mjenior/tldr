import os
import asyncio


class CompletionHandler:

    def __init__(self):
        pass

    def _scale_token(self, tokens):
        """Multiply the size of maximum output tokens allowed."""
        scaling = {"low": 0.5, "medium": 1.0, "high": 1.5}
        factor = scaling[self.token_scale]
        return int(tokens * factor)

    async def _async_single_completion(self, message, prompt_type, **kwargs):
        """Initialize and submit a single chat completion request"""

        # Fetch params
        instructions_dict = self.instructions[prompt_type]
        instructions = instructions_dict["system_instruction"]
        model = instructions_dict["model"]
        output_tokens = instructions_dict["max_output_tokens"]

        # Assemble messages object, adding query and context information
        messages = [
            {"role": "system", "content": instructions + self.user_query},
            {"role": "user", "content": message + self.added_context},
        ]

        # Rescale max tokens
        max_output_tokens = self._scale_token(output_tokens)

        # Run completion query
        try:
            completion = await self.async_client.chat.completions.create(
                messages=messages, model=model, max_tokens=max_output_tokens, **kwargs
            )
            # Extract and return text
            return completion.choices[0].message.content.strip()

        except openai.RateLimitError as e:
            # Handle hitting the rate limit silently
            pass

    async def async_multi_completions(self):
        """
        Runs multiple _async_single_completion calls concurrently using asyncio.gather, with retry on 429.
        """
        coroutine_tasks = []
        for ext, prompts in self.content.items():

            # Define correct summary to generate
            if ext == "pdf":
                prompt_type = "summarize_publication"
            elif ext == "md":
                prompt_type = "summarize_readme"
            else:
                prompt_type = "summarize_document"

            # Generate summaries for each prompt, handling rate limit errors
            for prompt in prompts:
                task = self._retry_on_429(
                    self._async_single_completion,
                    prompt=prompt,
                    prompt_type=prompt_type,
                )
                coroutine_tasks.append(task)

        return await asyncio.gather(*coroutine_tasks, return_exceptions=True)

    async def _retry_on_429(self, coro_fn, prompt, prompt_type, max_retries=5):
        """Retry summary generation on token rate limit per hour exceeded errors."""

        for attempt in range(1, max_retries + 1):
            try:
                return await coro_fn(prompt, prompt_type)
            except Exception as e:
                if hasattr(e, "status") and e.status == 429:
                    wait_time = random.uniform(*(1, 3)) * attempt
                    print(
                        f"Rate limit hit. Retrying in {wait_time:.2f}s (attempt {attempt})..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise  # Not a 429 error, re-raise

        raise RuntimeError("Max retries exceeded for task")

    def search_web(self, questions: str, context_size: str = "medium"):
        """Use web search tool to fill holes in current reading set"""

        instruction = "Use web search to answer the following questions as thoroughly as possible. Cite all sources.\n"
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview", "search_context_size": context_size}],
            input=instruction + questions,
        )

        return response.output_text

    def single_completion(self, message, prompt_type, **kwargs):
        """Single synchronous chat completion"""

        # Fetch params
        instructions_dict = self.instructions[prompt_type]
        instructions = instructions_dict["system_instruction"]
        model = instructions_dict["model"]
        output_tokens = instructions_dict["max_output_tokens"]

        # Assemble messages object
        messages = [
            {"role": "system", "content": instructions + self.user_query},
            {"role": "user", "content": message + self.added_context},
        ]

        # Rescale max tokens
        max_output_tokens = self._scale_token(output_tokens)

        # Get single chat response
        response = self.client.responses.create(
            input=messages, model=model, max_output_tokens=max_output_tokens, **kwargs
        )

        return response.output_text
