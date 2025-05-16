
import os
from openai import OpenAI

from .i_o import read_system_instructions

class SummaryJudge():
	"""Attempt to objectively score the quality of a summary from a highly technical resource."""
	def __init__(
    	self, 
    	summary, 
    	model='gpt-4o-mini',
    	iters=5, 
    	temp=0.6, 
    	rng=42, 
    	p=1.0,
    	api_key=None,
    	evals=rubric_str,
    	condensation=condense_str):

		# Assign attr
		self.summary = summary
        self.model = model
        self.iterations = iters
        self.evaluation_criteria = criteria
        self.temperature = temp
        self.random_seed = rng
        self.top_p = p

        # Assign instruction strings
        instructions = read_system_instructions()
        self.evaluation_criteria = instructions['rubric_instructions']['system_instruction']
        self.condense_iters = instructions['condense_instructions']['system_instruction']

		# Fetch API key
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided.")

        # Initialize OpenAI clients
		self.client = OpenAI(api_key=api_key)


	def score_summary(self, summary):
		""" """
		messages = [
	            {"role": "user", "content": summary},
	            {"role": "system", "content": self.evaluation_criteria},
	        ]

		response = self.client.chat.completions.create(
	            model=self.model,
	            messages=messages,
	            n=self.iterations,
	            seed=self.random_seed,
	            temperature=self.temperature,
	            top_p=self.top_p)

	        if self.iterations > 1:
	            message = self._condense_iterations(response)
	        else:
	            message = response.choices[0].message.content.strip()

	    return message


	def _condense_iterations(self, response, instructions=condense_str):
	        """Condenses multiple API responses into a single coherent response."""
	        self.evaluations = [r.message.content.strip() for r in response.choices]
	        responses = self._gen_iteration_str(self.evaluations)

	        messages = [{"role": "system", "content": condense_str}, 
	        	{"role": "user", "content": responses}]

	        condensed = self.client.chat.completions.create(
	            model=self.model,
	            seed=self.seed,
	            temperature=self.temperature,
	            top_p=self.top_p,
	            messages=messages)

	    return condensed.choices[0].message.content.strip()


	def _gen_iteration_str(self, responses):
		"""Format single string with response iteration text"""
		compiled_str = "\n\n".join(
		    [
		        "\n".join([f"Summary iteration: {i + 1}", responses[i]])
		        for i in range(len(responses))
		    ]
		)

		return compiled_str
