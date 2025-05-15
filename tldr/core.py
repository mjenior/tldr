
import os
import asyncio

from openai import OpenAI, AsyncOpenAI

from i_o import fetch_content, read_system_instructions, save_response_text
from completion import CompletionHandler


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
		self.content = fetch_content(search_directory)
		self.sources = sum([len(self.content[x]) for x in self.content.keys()])
		if self.verbose == True: print(f"\nIdentified {self.sources} references to include in summary.\n")

		# Read in system instructions
		self.instructions = read_system_instructions()


	def refine_user_query(self):
		if self.verbose == True: print('Refining user query...')
		new_query = self.completion(prompt=self.user_query, prompt_type='refine_prompt')
		addon = "Ensure that addressing the following user query is the central theme of your output text:\n{query}\n\n"
		save_response_text(new_query, label='new_query', output_dir=self.output_directory)

		self.user_query = addon + new_query


	async def summarize_resources(self):
		"""Generate component and synthesis summary text"""

		# Asynchronously summarize documents
		if self.verbose == True: print('Generating summaries from identified resources...')
		summaries = await self.async_completions(self.content)
		await self.async_client.close()
		
		# Join response strings
		all_summaries = ""
		for i in range(0, len(summaries)):
			all_summaries += f"Reference {i+1} Summary\n{summaries[i]}\n\n\n"
		save_response_text(all_summaries, label='summaries', output_dir=self.output_directory)

		return all_summaries


	def integrate_summaries(self):
		""" """
		if self.verbose == True: print('Generating integrated text...')

		user_prompt = self.user_query + self.all_summaries

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
		if self.verbose == True: print('\rResearching technical gaps in summary...')
		gap_questions = self.completion(message=self.content_synthesis, prompt_type='research_instructions')

		# Search web for to fill gaps
		filling_text = self.search_web(gap_questions)
		
		# Handle output
		research_text = f"""

		Gap questions:
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
		#if self.verbose == True: print(f"\n\n{polished}\n")
