
import os
import asyncio

from openai import AsyncOpenAI


class ChatCompletionHandler:
    def __init__(self, api_key=None):
        # Prefer API key from argument, fallback to environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
		self.client = AsyncOpenAI(api_key=self.api_key)


    async def async_completion(self, prompt, prompt_type, **kwargs):
		"""Initialize and submit a single chat completion request"""

		# Fetch params
		instructions_dict = self.instructions[prompt_type]
		instructions = instructions_dict['system_instruction']
		model = instructions_dict['model']
		max_output_tokens = instructions_dict['max_output_tokens']

		# Assemble messages object
		messages = [{"role": "user", "content": prompt}, {"role": "system", "content": instructions}]

		# Run completion query
		completion = await self.client.chat.completions.create(messages=messages, 
			model=model, max_completion_tokens=max_output_tokens)

		# Extract and return text
		return completion.choices[0].message.content.strip()


	async def async_multi_completion(self, prompt_dict):
	    """
	    Runs multiple async_chat_completion calls concurrently using asyncio.gather.
	    """
	    # Create a list of coroutine objects
	    coroutine_tasks = []
	    for ext in prompt_dict.keys():
	    	if ext == 'pdf':
	    		prompt_type = 'summarize_publication'
	    	else:
	    		prompt_type = 'summarize_document'

	    	for prompt in prompt_dict[ext]:
		        task = tldr_instance.async_completion(
		        	prompt=prompt, prompt_type=prompt_type)
		        coroutine_tasks.append(task)

	    # Gather all completion results
	    all_completions = await asyncio.gather(*coroutine_tasks, return_exceptions=True)

	    return all_completions


    def search_web(self, questions):
    	"""Use web search tool to fill holes in current reading set"""

    	# Initialize normal client
		client = OpenAI(api_key=self.api_key)

		# Get single chat response
		response = client.responses.create(
		    model="o4-mini",
		    tools=[{"type": "web_search_preview"}],
		    input=questions)

		# Retrun answers
		return response.output_text
