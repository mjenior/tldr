
import os
from openai import OpenAI

from .i_o import read_system_instructions

class SummaryJudge(CompletionHandler):
	"""Attempt to objectively score the quality of a summary from a highly technical resource."""
	def __init__(
		self, 
		summary, 
		model='gpt-4o-mini',
		iters=5, 
		temp=0.6, 
		rng=42, 
		top_p=1.0,
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
		self.top_p = top_p

		# Fetch API key abd initialize client
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise ValueError("OpenAI API key not provided.")
		self.client = OpenAI(api_key=api_key)

		# Reaad in system instructions
		instructions = read_system_instructions()


	def score_summary(self, summary):
		"""Generate multiple scoring responses for the provided summary text"""

		# Ensure replication
		iters = int(self.iterations) if self.iterations >= 3 else 3

		# Request repeated scoring text
		scoring = self.completion(
			message=summary, 
			prompt_type='rubric_instructions',
			n=iters,
			seed=self.random_seed,
			temperature=self.temperature,
			top_p=self.top_p)

		# Condense responses
		scoring = self._condense_iterations(scoring)

		return scoring


	def _condense_iterations(self, responses):
			"""Condenses multiple API responses into a single coherent message."""
			
			# Combine strings to demarcated single message
			responses = self._gen_iteration_str(responses)

			# Obtain final score text
			condensed = self.completion(
				message=responses, 
				prompt_type='rubric_instructions',
				n=self.iterations,
				seed=self.random_seed,
				temperature=self.temperature,
				top_p=self.top_p)

		return condensed


	def _gen_iteration_str(self, responses):
		"""Format single string with response iteration text"""
		compiled_str = "\n\n".join(
			["\n".join([f"Summary iteration: {i + 1}", responses[i].strip()])
				for i in range(len(responses))])

		return compiled_str
