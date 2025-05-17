
import os
import asyncio

from openai import OpenAI, AsyncOpenAI

from tldr.i_o import fetch_content, read_system_instructions, save_response_text
from tldr.completion import CompletionHandler


class TldrClass(CompletionHandler):
	"""
	Main TLDR class
	"""
	def __init__(
		self, 
		user_query = "",
		search_directory = ".",
		output_directory = ".",
		verbose=True,
		api_key=None,
		glyphs=False):

		self.user_query = user_query
		self.verbose = verbose
		self.output_directory = output_directory
		self.gap_research = ''
		self.glyph_synthesis = False

		# Set API key
		self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
		if not self.api_key:
			raise ValueError("OpenAI API key not provided.")

		# Initialize OpenAI clients
		self.client = OpenAI(api_key=self.api_key)
		self.async_client = AsyncOpenAI(api_key=self.api_key)

		# Read in files and contents
		if self.verbose == True: print('Searching input directory...')
		self.content = fetch_content(search_directory)
		self.sources = sum([len(self.content[x]) for x in self.content.keys()])
		if self.verbose == True: print(f"Identified {self.sources} references to include in synthesis.\n")

		# Read in system instructions
		self.instructions = read_system_instructions()

		# Lint user query
		if len(self.user_query) > 0: self._lint_user_query()


	def __call__(self, refine=False, research=False, polish=True):
		"""
		Callable interface to run the main TLDR pipeline.

		Arguments:
		- refine: whether to refine the user query
		- research: whether to search for missing background knowledge
		- polish: whether to finalize and polish the response
		"""

		async def _pipeline():
			if refine:
				self.refine_user_query()

			# Generate async summaries
			self.all_summaries = await self.summarize_resources()

			# Integrate summaries
			self.integrate_summaries()

			# Apply research if needed
			if research:
				self.apply_research()

			# Polish the final result
			self.polish_response()

		# Run the async pipeline
		asyncio.run(_pipeline())


	def _lint_user_query(self) -> None:
		"""
		Normalizes `self.user_query` to ensure it's a well-formed string query.
		"""
		current_query = self.user_query

		# 1. Handle list input or convert to string
		if isinstance(current_query, list):
			processed_query = ' '.join(map(str, current_query))
		elif isinstance(current_query, str):
			processed_query = current_query
		elif current_query is None:
			processed_query = ""
		else:
			processed_query = str(current_query)

		# Capitalize the first letter (if the string is not empty)
		if processed_query: # Check if the string is not empty
			processed_query = processed_query[0].upper() + processed_query[1:]
		
		# Ensure the query ends with a question mark (if the string is not empty)
		if processed_query and not processed_query.endswith('?'):
			processed_query += "?"
		elif not processed_query and self.user_query is not None:
			pass

		self.user_query = processed_query

		
	def refine_user_query(self):
		if self.verbose == True: print('Refining user query...')
		new_query = self.completion(message=self.user_query, prompt_type='refine_prompt')
		addon = "Ensure that addressing the following user query is the central theme of your output text:\n{query}\n\n"
		save_response_text(new_query, label='new_query', output_dir=self.output_directory)
		if self.verbose == True: print('Refined query:\n{new_query}\n')
		self.user_query = addon + new_query


	async def summarize_resources(self):
		"""Generate component and synthesis summary text"""

		# Asynchronously summarize documents
		if self.verbose == True: print('Generating summaries for selected resources...')
		summaries = await self.async_completions(self.content)
		await self.async_client.close()
		
		# Join response strings
		all_summaries = []
		for i in range(0, len(summaries)):
			all_summaries.append(f"Reference {i+1} Summary\n{summaries[i]}\n")
		save_response_text(all_summaries, label='summary', output_dir=self.output_directory)

		return all_summaries


	def integrate_summaries(self):
		""" """
		if self.verbose == True: print('Synthesizing integrated summary...')

		user_prompt = self.user_query + "\n\n".join(self.all_summaries)

		if self.glyph_synthesis == True:

			# Generate glyph-formatted prompt
			glyph_synthesis = self.completion(message=user_prompt, prompt_type='glyph_prompt')
			save_response_text(glyph_synthesis, label='glyph_synthesis', output_dir=self.output_directory)

			# Summarize documents based on glyph summary
			synthesis = self.completion(message=glyph_synthesis, prompt_type='interpret_glyphs')
		else:
			synthesis = self.completion(message=user_prompt, prompt_type='executive_summary')

		save_response_text(synthesis, label='synthesis', output_dir=self.output_directory)

		self.content_synthesis = synthesis


	def apply_research(self):
		"""Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""
		
		# Identify gaps in understanding
		if self.verbose == True: print('\rIdentifying technical gaps in summary...')
		gap_questions = self.completion(message=self.content_synthesis, prompt_type='research_instructions')

		# Search web for to fill gaps
		if self.verbose == True: print('\rResearching gaps in important background...')
		filling_text = self.search_web(gap_questions)
		
		# Handle output
		research_text = f"""Gap questions:
{gap_questions}

Answers:
{filling_text}
"""
		save_response_text(research_text, label='research', output_dir=self.output_directory)

		self.gap_research = research_text


	def polish_response(self, output_type: str = 'default'):
		"""Refine final response text"""
		if self.verbose == True: print('Finalizing response text...')

		# Select tone
		tone = 'polishing_instructions' if output_type == 'default' else 'modified_polishing'
		response_text = self.user_query + self.content_synthesis + self.gap_research

		# Run completion and save final response text
		polished = self.completion(message=response_text, prompt_type=tone)
		save_response_text(polished, label='final', output_dir=self.output_directory)

		# Print final answer to terminal
		if self.verbose == True: print(f"\n\n{polished}\n")
