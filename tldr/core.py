
import os

from openai import AsyncOpenAI

from .read import fetch_content
from .utils import system_instructions
from .completion import ChatCompletionHandler


class TldrClass(ChatCompletionHandler):
    """
	Main TLDR class
    """
    def __init__(self, query=None, search_directory = "."):

    	# Confirm environment API key
		self.api_key = os.getenv("OPENAI_API_KEY")
		if self.api_key is None:
		    raise EnvironmentError("OPENAI_API_KEY environment variable not found!")

		# Initialize async OpenAI client
		self.client = AsyncOpenAI(api_key=api_key)

		# Read in files and contents
		self.content = fetch_content(search_directory)

		# Read in system instructions and pair with content
		self.instructions = system_instructions()
        self.filetype_instructions = self._pair_instructions_with_filetypes()

		# Extend user query
		if query is not None:
			self.query = self.chat_completion(query, self.instructions['refine_prompt']['system_instruction'])
		else:
			self.query = query


    def _pair_instructions_with_filetypes(self) -> dict:
        """
        Pairs file types found in self.files with relevant system instructions.
        """
        doc_summary = self.instructions["summarize_document"]["system_instruction"]
        pub_summary = self.instructions["summarize_publication"]["system_instruction"]

        # Iterate through the file extensions we actually content for
        paired_instructions = {}
        for ext in self.content.keys():
        	for text in self.content[ext]:
        		paired_instructions[text] = pub_summary if key == 'pdf' else doc_summary

        return paired_instructions


	def save_response_text(data_str: str, filename: str, errors: str = 'strict', chunk_size: int = 1024*1024):
	    """
	    Saves a large string variable to a text file.
	    """
	    try:
	        with open(filename, 'w', encoding='utf-8', errors=errors) as f:
	            for i in range(0, len(data_str), chunk_size):
	                f.write(data_string[i:i + chunk_size])
	        print(f"Successfully saved data to {filename}")
	    except IOError as e:
	        print(f"Error writing to file {filename}: {e}")
	        raise  # Re-raise the exception after printing
	    except Exception as e:
	        print(f" An unexpected error occurred: {e}")
	        raise # Re-raise the exception
