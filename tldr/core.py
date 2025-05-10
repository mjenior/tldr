
import os

from openai import AsyncOpenAI

from .read import fetch_content
from .utils import system_instructions, save_response_text
from .completion import ChatCompletionHandler


class TldrClass(ChatCompletionHandler):
    """
	Main TLDR class
    """
    def __init__(
    	self, 
    	query=None, 
    	search_directory = ".",
    	verbose=True):

    	self.verbose = verbose
    	self.output_directory = output_directory

		# Read in files and contents
		self.content = fetch_content(search_directory)
		self.sources = sum([len(self.content[x]) for x in self.content.keys()])

		# Read in system instructions
		self.instructions = system_instructions()

		# Extend user query
		if query is not None:
			query = self.async_completion(prompt=query, prompt_type='refine_prompt')
			addon = "\n\nAllow the following user query to direct points of focus in your summary:\n\n"
			self.query = addon + query
		else:
			self.query = ""


	def summarize_resources(self):
		"""Generate component and synthesis summary text"""

		# Summarize documents
	    if self.verbose == True:
	        print(f'Generating summary from {self.sources} resources...')
	    summaries = await self.async_multi_completion(self.content)
	    summary = "\n\n".join(summaries)
	    save_response_text(summary, label='summaries', output_dir=self.output_directory)

	    # Generate synthesis
	    prompt = f"{summary}\n\n{self.query}"
	    glyph_synthesis = await self.async_completion(prompt=prompt, prompt_type='glyph_prompt')
	    synthesis = await self.async_completion(prompt=glyph_synthesis, prompt_type='interpret_glyphs')
	    save_response_text(synthesis, label='synthesis', output_dir=self.output_directory)

	    return synthesis


    def apply_research(self, summary):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""
        
        # Identify gaps in understanding
        if self.verbose == True:
            print('Researching technical gaps in summary...')
        gap_questions = await self.async_completion(prompt=summary, prompt_type='research_instructions')

        # Search web for to fill gaps
        filling_text = self.search_web(gap_questions)
        research_text = f"{gap_questions}\n\n{filling_text}"
        save_response_text(research_text, label='research', output_dir=self.output_directory)

        # Add findings to previous summary text
        return f"{summary}\n\n{filling_text}"


    def polish_response(self, response, query, output_type: str = 'default'):
    	""" """
    	if self.verbose == True:
            print('Finalizing response text...\n\n')

        if query != "":
        	query = "Ensure addressing the following user query is the central theme of the text below it:\n{query}\n\n" + 

        tone = 'polishing_instructions' if output_type == 'default' else 'modified_polishing'

        # Run completion and save final response text
    	polished = await self.async_completion(prompt=query+response, prompt_type=tone)
    	save_response_text(polished, label='final', output_dir=self.output_directory)

    	# Print final answer to terminal
    	if self.verbose == True:
    		print(polished)

    	return polished

