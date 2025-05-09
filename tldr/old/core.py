
import os
import re
import sys
import time
import string
import requests
from datetime import datetime
from collections import defaultdict

from openai import OpenAI

import fitz  # PyMuPDF
import io
from PIL import Image # Pillow library for image handling


class CreateAgent:


    def __init__(
        self,
        verbose = True,
        silent = False,
        refine = False,
        reasoning = False,
        scandirs = False,
        model = "gpt-4o",
        seed = "t634e``R75T86979UYIUHGVCXZ",
        temperature = 0.7,
        top_p = 1.0,
        message_limit = 20):
        """
        Initialize the handler with default or provided values.
        """
        self.logging = True
        self.save_code = True

        self.verbose = verbose
        self.silent = silent
        self.refine_prompt = refine
        self.reasoning = reasoning
        self.scan_dirs = scandirs
        self.model = model
        self.role = role
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.dimensions = dimensions
        self.quality = quality
        self.message_limit = message_limit

        # Check user input types
        self._validate_types()

        # Agent-specific thread params
        global thread
        self.thread_id = thread.id
        thread.message_limit = message_limit

        # Update token counters
        global total_tokens
        self.cost = {"prompt": 0.0, "completion": 0.0}
        self.tokens = {"prompt": 0, "completion": 0}
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}
        
        # Validdate specific hyperparams
        self.seed = self.seed if isinstance(self.seed, int) else self._string_to_binary(self.seed)
        self.temperature, self.top_p = self._validate_probability_params(self.temperature, self.top_p)
        
        # Validate user inputs
        self._prepare_system_role(role)
        self._validate_model_selection(model)
        if self.model in ["dall-e-2", "dall-e-3"]:
            self._validate_image_params(dimensions, quality)
        self._create_new_agent(interpreter=self.save_code)

        # Initialize reporting and related vars
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.prefix = f"{self.label}.{self.model.replace('-', '_')}.{self.timestamp}"        
        if self.logging: self._setup_logging()
        self._log_and_print(self.status(), False, self.logging)

    def _validate_types(self):
        """
        Validates the types of the instance attributes for CreateAgent.

        Raises:
            TypeError: If any attribute has an incorrect type.
            ValueError: If any integer attribute is not positive.
        """
        expected_types = {
            'logging': bool,
            'verbose': bool,
            'silent': bool,
            'refine_prompt': bool,
            'reasoning': bool,
            'save_code': bool,
            'scan_dirs': bool,
            'new_thread': bool,
            'model': str,
            'role': str,
            'seed': (int, str),  # seed can be either int or str
            'temperature': float,
            'top_p': float,
            'dimensions': str,
            'quality': str,
            'message_limit': int
        }

        for attr_name, expected_type in expected_types.items():
            value = getattr(self, attr_name, None)  # Get the attribute value from self
            if isinstance(expected_type, tuple):
                # Check if value matches any expected type in the tuple
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}")
            else:
                # Check if value matches the expected type
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}")
            
            # Check if integer-type values are positive
            if expected_type == int and value <= 0:
                raise ValueError(f"{attr_name} must be a positive integer, got {value}")

    def _setup_logging(self):
        """
        Prepare logging setup.
        """
        self.log_text = []
        self.log_file = f"logs/{self.prefix}.transcript.log"
        os.makedirs("logs", exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("New session initiated.\n")

    def _validate_probability_params(self, temp, topp):
        """Ensure temperature and top_p are valid"""
        # Acceptable ranges
        if temp < 0.0 or temp > 2.0:
            temp = 0.7
        if topp < 0.0 or topp > 2.0:
            topp = 1.0

        # Only one variable is changed at a time
        if temp != 0.7 and topp != 1.0:
            topp = 1.0

        return temp, topp

    def _prepare_query_text(self, prompt_text):
        """
        Prepares the query, including prompt modifications and image handling.
        """
        self.prompt = prompt_text

        # Identifies files to be read in
        files = self._find_existing_files()
        for f in files:
            self.prompt += "\n\n" + self._read_file_contents(f)
        if self.scan_dirs == True:
            paths = self._find_existing_paths()
            for d in paths:
                self.prompt += "\n\n" + self._scan_directory(d)

        # Refine prompt if required
        if self.refine_prompt:
            self._log_and_print(
                "\nAgent using gpt-4o-mini to optimize initial user request...\n", True, self.logging)
            self.prompt = self._refine_user_prompt(self.prompt)


    def _prepare_system_role(self, input_role):
        """Prepares system role text."""

        # Selects the role based on user input or defaults.
        if input_role.lower() in roleDict:
            self.label = input_role.lower()
            builtin = roleDict[input_role.lower()]
            self.role = builtin["prompt"]
            self.role_name = builtin["name"]
        elif input_role.lower() in ["user", ""]:
            self.role = "user"
            self.label = "default"
            self.role_name = "Default User"
        else:
            self.role_name, self.role = self._refine_custom_role(input_role)
            self.label = "custom"

        # Add chain of thought reporting
        if self.reasoning:
            self.role += modifierDict["cot"]



    def status(self):
        """Generate status message."""
        statusStr = f"""
Agent parameters:
    Model: {self.model}
    Role: {self.role_name}
    
    Chain-of-thought: {self.reasoning}
    Prompt refinement: {self.refine_prompt}
    Subdirectory scanning: {self.scan_dirs}
    Text logging: {self.logging}
    Verbose StdOut: {self.verbose}
    Code snippet detection: {self.save_code}

    Time stamp: {self.timestamp}
    Seed: {self.seed}
    Assistant ID: {self.agent}
    Thread ID: {thread.id}
    Requests in current thread: {thread.current_thread_calls}
    """
        self._log_and_print(statusStr, True, self.logging)

        # Token usage report
        self._token_report()
        
        # $$$ report
        self._cost_report()

        # Thread report
        self._thread_report()

    def new_thread(self, context=None):
        """Start a new thread with only the current agent and adds previous context if needed."""
        global thread
        global client
        thread = client.beta.threads.create()
        thread.current_thread_calls = 0
        thread.message_limit = self.message_limit

        # Add previous context
        if context:
            previous_context = client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=context)

        client.thread_ids.append(thread.id)
        self.thread_id = thread.id

        # Report
        self._log_and_print(f"New thread with previous context added to current agent: {self.thread_id}\n", 
            self.verbose, self.logging)

    def request(self, prompt=''):
        """Submits the query to OpenAIs API and processes the response."""
        # Checks for last system response is not prompt provided
        if prompt == '':
            try:
                prompt = self.last_message
            except Exception as e:
                raise ValueError(f"No existing messages found in thread: {e}")

        # Update user prompt 
        self._prepare_query_text(prompt)
        self._log_and_print(
            f"\n{self.role_name} using {self.model} to process updated conversation thread...\n",
                True, self.logging)

        # Check current scope thread
        if thread.current_thread_calls >= thread.message_limit:
            self._log_and_print(f"Reached end of current thread limit.\n", self.verbose, False)
            summary = self.summarize_thread()
            self.new_thread("The following is a summary of a ongoing conversation with a user and an AI assistant:\n" + summary)

    def _init_chat_completion(self, prompt, model='gpt-4o-mini', role='user', seed=42, temp=0.7, top_p=1.0):
        """Initialize and submit a single chat completion request"""
        message = [{"role": "user", "content": prompt}, {"role": "system", "content": role}]

        completion = client.chat.completions.create(
            model=model, messages=message, 
            seed=seed, temperature=temp, top_p=top_p)
        self._update_token_count(completion)
        self._calculate_cost()

        return completion

    def summarize_thread(self):
        """Summarize current conversation history for future context parsing."""
        self._log_and_print(f"Agent using gpt-4o-mini to summarize current thread...\n", self.verbose, False)

        # Get all thread messages
        all_messages = self._get_thread_messages()

        # Generate concise summary
        summary_prompt = modifierDict['summarize'] + "\n\n" + all_messages
        summarized = self._init_chat_completion(prompt=summary_prompt, seed=self.seed)

        return summarized.choices[0].message.content.strip()

    def _get_thread_messages(self):
        """Fetches all messages from a thread in order and returns them as a text block."""
        messages = client.beta.threads.messages.list(thread_id=self.thread_id)
        sorted_messages = sorted(messages.data, key=lambda msg: msg.created_at)
        conversation = [x.content[0].text.value.strip() for x in sorted_messages]

        return "\n\n".join(conversation)

    def _handle_text_request(self):
        """Processes text-based responses from OpenAIs chat models."""
        self.last_message = self._run_thread_request()
        self._update_token_count(self.run_status)
        self._calculate_cost()
        self._log_and_print(self.last_message, True, self.logging)

        # Extract code snippets
        code_snippets = self._extract_code_snippets()
        if self.save_code and len(code_snippets) > 0:
            self.code_files = []
            reportStr = "\nExtracted code saved to:\n"
            for lang in code_snippets.keys():
                code = code_snippets[lang]
                objects = select_object_name(code, lang)
                file_name = f"{self._select_largest_object(code, objects)}.{self.timestamp}{extDict.get(lang, f'.{lang}')}".lstrip("_.")
                reportStr += f"\t{file_name}\n"
                self._write_script(code, file_name)

            self._log_and_print(reportStr, True, self.logging)



    def _update_token_count(self, response_obj):
        """Updates token count for prompt and completion."""
        global total_tokens
        total_tokens[self.model]["prompt"] += response_obj.usage.prompt_tokens
        total_tokens[self.model]["completion"] += response_obj.usage.completion_tokens
        # Agent-specific counts
        self.tokens["prompt"] += response_obj.usage.prompt_tokens
        self.tokens["completion"] += response_obj.usage.completion_tokens

    def _token_report(self):
        """Generates session token report."""
        allTokensStr = ""
        for x in total_tokens.keys():
            allTokensStr += f"{x}: Input = {total_tokens[x]['prompt']}; Completion = {total_tokens[x]['completion']}\n"

        tokenStr = f"""Overall session tokens:
    {allTokensStr}
    Current agent tokens: 
        Input: {self.tokens['prompt']}
        Output: {self.tokens['completion']}
"""
        self._log_and_print(tokenStr, True, self.logging)

    def _thread_report(self):
        """Report active threads from current session"""

        ids = '\n\t'.join(client.thread_ids)
        threadStr = f"""Current session threads:
    {ids}
"""
        self._log_and_print(threadStr, True, self.logging)

    def _calculate_cost(self, dec=5):
        """Calculates approximate cost (USD) of LLM tokens generated to a given decimal place"""
        global total_cost

        # As of January 24, 2025
        rates = {
            "gpt-4o": (2.5, 10),
            "gpt-4o-mini": (0.150, 0.600),
            "o1-mini": (3, 12),
            "o1-preview": (15, 60),
            "dall-e-3": (2.5, 0.040),
            "dall-e-2": (2.5, 0.040),
        }
        if self.model in rates:
            prompt_rate, completion_rate = rates.get(self.model)
            prompt_cost = round((self.tokens["prompt"] * prompt_rate) / 1e6, dec)
            completion_cost = round((self.tokens["completion"] * completion_rate) / 1e6, dec)
        else:
            prompt_cost = completion_cost = 0.0

        total_cost += round(prompt_cost + completion_cost, dec)
        self.cost["prompt"] += prompt_cost
        self.cost["completion"] += completion_cost

    def _cost_report(self, dec=5):
        """Generates session cost report."""
        
        costStr = f"""Overall session cost: ${round(total_cost, dec)}

    Current agent using: {self.model}
        Subtotal: ${round(self.cost['prompt'] + self.cost['completion'], dec)}
        Input: ${self.cost['prompt']}
        Output: ${self.cost['completion']}
"""     
        self._log_and_print(costStr, True, self.logging)

    def _refine_user_prompt(self, old_prompt):
        """Refines an LLM prompt using specified rewrite actions."""
        updated_prompt = old_prompt
        if self.refine_prompt == True:
            actions = set(["expand", "amplify"])
            actions |= set(
                re.sub(r"[^\w\s]", "", word).lower()
                for word in old_prompt.split()
                if word.lower() in refineDict
            )
            action_str = "\n".join(refineDict[a] for a in actions) + "\n\n"
            updated_prompt = modifierDict["refine"] + action_str + old_prompt

        refined = self._init_chat_completion(
            prompt=updated_prompt, 
            role=self.role,
            seed=self.seed, 
            temp=self.temperature, 
            top_p=self.top_p)

        if self.responses > 1:
            new_prompt = self._condense_responses(refined)
        else:
            new_prompt = refined.choices[0].message.content.strip()

        self._log_and_print(
            f"Refined query prompt:\n{new_prompt}", self.verbose, self.logging)

        return new_prompt

    def _log_and_print(self, message, verb=True, log=True):
        """Logs and prints the provided message if verbose."""
        if message:
            if verb == True and self.silent == False:
                print(message)
            if log == True:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")

    @staticmethod
    def _string_to_binary(input_string):
        """Create a binary-like variable from a string for use a random seed"""
        # Convert all characters in a str to ASCII values and then to 8-bit binary
        binary = ''.join([format(ord(char), "08b") for char in input_string])
        # Constrain length
        return int(binary[0 : len(str(sys.maxsize))])


    def _create_new_agent(self, interpreter=False):
        """
        Creates a new assistant based on user-defined parameters

        Args:
            interpreter (bool): Whether to enable the code interpreter tool.

        Returns:
            New assistant assistant class instance
        """
        try:
            agent = client.beta.assistants.create(
                name=self.role_name,
                instructions=self.role,
                model=self.model,
                tools=[{"type": "code_interpreter"}] if interpreter == True else [])
            self.agent = agent.id
        except Exception as e:
            raise RuntimeError(f"Failed to create assistant: {e}")

    def _run_thread_request(self) -> str:
        """
        Sends a user prompt to an existing thread, runs the assistant, 
        and retrieves the response if successful.
        
        Returns:
            str: The text response from the assistant.
        
        Raises:
            ValueError: If the assistant fails to generate a response.
        """
        # Adds user prompt to existing thread.
        try:
            new_message = client.beta.threads.messages.create(
                thread_id=self.thread_id, role="user", content=self.prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to create message: {e}")

        # Run the assistant on the thread
        current_run = client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.agent)

        # Wait for completion and retrieve responses
        while True:
            self.run_status = client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=current_run.id)
            if self.run_status.status in ["completed", "failed"]:
                break
            else:
                time.sleep(1)  # Wait before polling again

        if self.run_status.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=self.thread_id)
            if messages.data:  # Check if messages list is not empty
                return messages.data[0].content[0].text.value
            else:
                raise ValueError("No messages found in the thread.")
        else:
            raise ValueError("Assistant failed to generate a response.")






    def extract_figures_from_pdf(self, pdf_path: str,
                                 output_dir: str = "extracted_figures",
                                 min_width: int = 50,
                                 min_height: int = 50,
                                 output_format: str = "png") -> list[str]:
        """
        Extracts embedded bitmap images (potential figures) from a PDF file.

        It saves images that meet minimum dimension criteria to an output directory.
        Note: This function primarily extracts raster images (like PNG, JPG)
        embedded in the PDF. It may not reliably extract figures drawn using
        vector graphics commands unless they are represented as image objects.
        Identification relies on image dimensions, not semantic understanding
        (e.g., detecting captions like "Figure 1").

        Args:
            pdf_path: Path to the input PDF file.
            output_dir: Directory where extracted images will be saved.
                        It will be created if it doesn't exist.
            min_width: Minimum width (in pixels) for an image to be considered a figure.
                       Helps filter out small icons, logos, or noise.
            min_height: Minimum height (in pixels) for an image to be considered a figure.
                        Helps filter out small icons, logos, or noise.
            output_format: The format to save the images in (e.g., "png", "jpeg").
                           Pillow must support this format.

        Returns:
            A list of file paths where the extracted images were saved.
            Returns an empty list if no suitable images are found or if an error occurs.

        Raises:
            FileNotFoundError: If the pdf_path does not exist.
            Exception: For errors during PDF processing or image saving.
        """
        saved_image_paths = []

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Error: PDF file not found at '{pdf_path}'")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing PDF: {pdf_path}")
        print(f"Saving figures to: {output_dir}")
        print(f"Minimum dimensions (WxH): {min_width}x{min_height} pixels")
        print(f"Output format: {output_format}")

        try:
            # Open the PDF
            doc = fitz.open(pdf_path)

            image_count = 0
            # Iterate through each page
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                # Get list of images on the page.
                # get_images(full=True) provides more details including xref
                image_list = page.get_images(full=True)

                if image_list:
                    print(f"Found {len(image_list)} image object(s) on page {page_index + 1}")

                # Iterate through image objects on the page
                for img_index, img_info in enumerate(image_list):
                    # img_info is a tuple, the first element (xref) is the image object reference
                    xref = img_info[0]

                    try:
                        # Extract the raw image dictionary
                        base_image = doc.extract_image(xref)
                    except Exception as e:
                        print(f"  Warning: Could not extract image object with xref {xref} on page {page_index + 1}. Skipping. Error: {e}")
                        continue # Skip to the next image object

                    if not base_image:
                        print(f"  Warning: Got empty base_image for xref {xref} on page {page_index + 1}. Skipping.")
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Check if image dimensions meet the minimum criteria
                    if width >= min_width and height >= min_height:
                        image_count += 1
                        # Convert potentially non-standard formats using Pillow for wider compatibility
                        try:
                            img = Image.open(io.BytesIO(image_bytes))

                            # Define output filename
                            output_filename = f"page{page_index + 1}_fig{image_count}.{output_format.lower()}"
                            output_filepath = os.path.join(output_dir, output_filename)

                            # Save the image in the desired format
                            img.save(output_filepath, format=output_format.upper())
                            saved_image_paths.append(output_filepath)
                            print(f"  Saved: {output_filepath} (Dimensions: {width}x{height})")

                        except Exception as e:
                            print(f"  Warning: Failed to process/save image from xref {xref} (page {page_index + 1}) using Pillow. Skipping. Error: {e}")
                    else:
                        print(f"  Skipped: Image object {img_index + 1} on page {page_index + 1} (Dimensions: {width}x{height} too small)")

            # Close the document
            doc.close()

        except fitz.fitz.FileNotFoundError: # PyMuPDF specific file error
             raise FileNotFoundError(f"Error: PDF file not found or cannot be opened at '{pdf_path}'")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Optionally re-raise the exception if needed
            # raise e
            return [] # Return empty list on error

        if not saved_image_paths:
            print("No images meeting the criteria were found or extracted.")
        else:
            print(f"\nExtraction complete. Found and saved {len(saved_image_paths)} potential figures.")

        return saved_image_paths
