
import os
from openai import OpenAI

from .completion import CompletionHandler
from .i_o import read_system_instructions, read_file_content

class SummaryJudge(CompletionHandler):
	"""Attempt to objectively score the quality of a summary from a highly technical resource."""
	def __init__(
		self, 
		content_path,
		summary, 
		model='gpt-4o-mini',
		verbose=True,
		iters=5, 
		temp=0.6, 
		rng=42, 
		top_p=1.0,
		api_key=None,
		evals=rubric_str,
		condensation=condense_str):

		# Assign attr
		self.model = model
		self.verbose = verbose
		self.iterations = iters
		self.evaluation_criteria = criteria
		self.temperature = temp
		self.random_seed = rng
		self.top_p = top_p

		# Read target content and summary files
		self.content = read_file_content(content_path)
		self.summary = read_file_content(summary_path)

		# Fetch API key abd initialize client
		api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise ValueError("OpenAI API key not provided.")
		self.client = OpenAI(api_key=api_key)

		# Read in system instructions
		self.instructions = read_system_instructions()


	def _update_request(self):
		"""Form combined content and summary string"""

		full_message = """
		The following content is the original target text of expert summary generating agent:
		{self.content}

		Below here is the generated summary you are to score:
		{self.summary}
		"""

		return full_message


	def generate_scoring(self):
		"""Generate multiple scoring responses for the provided summary text"""

		# Ensure replication
		iters = int(self.iterations) if self.iterations >= 3 else 3

		# Combine target and summary texts
		message = self._update_request()

		# Request repeated scoring text
		if self.verbose == True:
			print("Generating summary evaluation iterations...")
		scoring = self.completion(
			message=self.summary, 
			prompt_type='rubric_instructions',
			n=iters,
			seed=self.random_seed,
			temperature=self.temperature,
			top_p=self.top_p)

		# Condense responses
		if self.verbose == True:
			print("Condensing final evaluation text...")
		scoring = self._condense_iterations(scoring)

		return scoring


	def _condense_iterations(self, responses):
		"""Condenses multiple API responses into a single coherent message."""
		
		# Combine strings into demarcated single message
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
