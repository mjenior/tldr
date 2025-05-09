
import os
import asyncio

from openai import OpenAI


class ChatCompletionHandler:
    def __init__(self, api_key=None):
        # Prefer API key from argument, fallback to environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")


    def chat_completion(self, prompt, instructions, **kwargs):
		"""Initialize and submit a single chat completion request"""

		# Initialize normal client
		client = OpenAI(api_key=self.api_key)

		# Assemble messages object
		messages = [{"role": "user", "content": prompt}, {"role": "system", "content": instructions}]

	    # Construct parameters dictionary, allowing kwargs to override defaults
	    params = {
	        'model': kwargs.pop('temperature', 'gpt-4o-mini'),
	        'messages': messages,
	        'temperature': kwargs.pop('temperature', 0.7), 
	        'top_p': kwargs.pop('top_p', 1.0), 
	        'seed': kwargs.pop('seed', 42), 
	        'max_tokens': kwargs.pop('max_tokens', 2048), 
	        **kwargs}

		# Run completion query
		completion = client.chat.completions.create(**params)

		# Extract and return text
		return completion.choices[0].message.content.strip()


    async def async_completion(
        self,
        content_dict: dict,
        **kwargs # For any other arguments like frequency_penalty etc.
    ) -> str:
        """
        Initialize and submit a single asynchronous chat completion request.
        This is a helper for the main async function.
        """
        for content in content_dict.keys():

        try:
            # Assemble messages object
            messages = [
                {"role": "user", "content": prompt},
                {"role": "system", "content": instructions}
            ]

            # Construct parameters dictionary
            params = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'top_p': top_p,
                'seed': seed,
                'max_tokens': max_tokens,
                **kwargs # Add any extra kwargs passed from the calling function
            }

            # Run completion query asynchronously
            completion = await self.async_client.chat.completions.create(**params)

            # Extract and return text
            return completion.choices[0].message.content.strip()

        except Exception as e:
            # Handle potential errors for this specific completion
            print(f"Error processing prompt: '{prompt[:50]}...' - {e}")
            return f"Error: {e}" # Or return None, or raise the exception


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





	async def async_multi_completion(self, data_for_completions: list):
	    """
	    Runs multiple async_chat_completion calls concurrently using asyncio.gather.
	    """
	    # Create a list of coroutine objects
	    coroutine_tasks = []
	    for text in data_for_completions.keys():
	        task = tldr_instance.async_chat_completion(
	        	prompt=text, instructions=data_for_completions[text])
	        coroutine_tasks.append(task)

	    # Gather all completion results
	    all_summaries = await asyncio.gather(*coroutine_tasks, return_exceptions=True)

	    # You might want to process the results list, checking for errors if return_exceptions=True
	    processed_results = []
	    for i, result in enumerate(all_summaries):
	        if isinstance(result, Exception):
	            processed_results.append(f"Task Failed: {result}")
	        else:
	            processed_results.append(result) # Add the successful result

	    return processed_results



