import os
import yaml
from datetime import datetime

from docx import Document
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

class FileHandler:
    def create_timestamp(self):
        """Generate a timestamp string (e.g., 20231027_103000)"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_output_path(self):
        """Set up where to write output summaries"""

        # Create full path for intermediate files
        if self.run_tag is not None:
            output_path = f"{self.run_tag}_files"
        else:
            output_path = f"tldr.{self.create_timestamp()}_files"
        os.makedirs(output_path, exist_ok=True)
        self.output_directory = output_path

    def fetch_content(
        self,
        user_files: list = None,
        search_dir: str = ".",
        recursive: bool = False,
    ):
        """
        Find files and read in their contents.
        Returns a list of content strings.
        """
        files = self._find_readable_files(
            infiles=user_files, directory=search_dir, recursive=recursive
        )
        content = []
        for ext in files.keys():
            content += self.read_file_content(files[ext], ext)
        return content

    def _find_readable_files(
        self, infiles: list = None, directory: str = ".", recursive: bool = False
    ) -> dict:
        """
        Scan for readable text files.
        Args:
            infiles: Optional list of specific files to process.
            directory: The path to the directory to scan. Defaults to the current directory.
            recursive: Whether to search recursively in subdirectories.
        Returns:
            A dictionary with file extensions as keys and lists of file paths as values.
        """
        readable_files_by_type = {"pdf": [], "docx": [], "html": [], "txt": [], "md": []}

        if infiles is not None:
            for file_path_item in infiles:
                ext = os.path.splitext(file_path_item)[1].lower()
                self._update_file_dictionary(readable_files_by_type, file_path_item, ext)
        else:
            if not recursive:
                for filename in os.listdir(directory):
                    if ".tldr." in filename:
                        continue
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        ext = os.path.splitext(filename)[1].lower()
                        self._update_file_dictionary(readable_files_by_type, filepath, ext)
            else:
                for root, _, files_in_dir in os.walk(directory):
                    for filename in files_in_dir:
                        if ".tldr." in filename:
                            continue
                        filepath = os.path.join(root, filename)
                        ext = os.path.splitext(filename)[1].lower()
                        self._update_file_dictionary(readable_files_by_type, filepath, ext)
        return {k: v for k, v in readable_files_by_type.items() if v}

    def _update_file_dictionary(self, file_dict, file_path, file_ext):
        """Add file path to dictionary if readable and of a supported type."""
        try:
            if file_ext == ".pdf":
                reader = PdfReader(file_path)
                if reader.pages and any(page.extract_text() for page in reader.pages if page.extract_text()): 
                    file_dict["pdf"].append(file_path)
            elif file_ext == ".docx":
                doc = Document(file_path)
                if any(para.text.strip() for para in doc.paragraphs):
                    file_dict["docx"].append(file_path)
            elif file_ext in [".html", ".htm"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)  # Try reading a small chunk to check readability
                file_dict["html"].append(file_path)
            elif file_ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                file_dict["md"].append(file_path)
            elif file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                file_dict["txt"].append(file_path)
        except Exception as e:  # pylint: disable=broad-except
            # Silently ignore files that can't be processed or raise an error
            pass 

    def read_file_content(self, filelist, ext=None):
        """
        Reads the main body text from given file paths and returns as strings.
        """
        if isinstance(filelist, str):
            filelist = [filelist]

        content_list = []
        for filepath in filelist:
            current_ext = ext
            if current_ext is None:
                current_ext = os.path.splitext(filepath)[1].lower().lstrip('.')
            
            try:
                if current_ext == "pdf":
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text: 
                            text += page_text + "\n"
                elif current_ext in ["doc", "docx"]:
                    doc = Document(filepath)
                    text = "\n".join([para.text for para in doc.paragraphs])
                elif current_ext in ["htm", "html"]:
                    with open(filepath, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text(separator="\n")
                elif current_ext in ["txt", "md"]:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                
                content_list.append(text.strip())

            except Exception as e: # pylint: disable=broad-except
                print(f"Error reading file '{filepath}': {e}")
                continue
            
        return content_list

    def read_system_instructions(self, file_path: str = "instructions.yaml") -> dict:
        """
        Reads a YAML file and returns its content as a Python dictionary.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        instructions_path = os.path.join(base_dir, file_path)
        instructions = {}
        try:
            with open(instructions_path, "r", encoding="utf-8") as file:
                instructions = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Instructions file not found: '{instructions_path}'")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file '{instructions_path}': {e}")
        except Exception as e: # pylint: disable=broad-except
            print(f"An unexpected error occurred while reading '{instructions_path}': {e}")
        return instructions

    def save_response_text(
        self,
        out_data: str,
        label: str = "response",
        output_dir: str = ".",
        idx: int = 1,
        chunk_size: int = 1024 * 1024, 
        errors: str = "strict"
    ) -> str:
        """
        Saves a string variable to a text file with a dynamic filename.
        This method now incorporates the logic of the former _save_summary_txt function.
        Returns the full path to the saved file, or an empty string on error.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.create_timestamp()
        if label == "summary":
            filename = f"{label}.{idx}.tldr.{timestamp}.txt"
        else:
            filename = f"{label}.tldr.{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8", errors=errors) as f:
                for i in range(0, len(out_data), chunk_size):
                    f.write(out_data[i:i + chunk_size])
            return filepath
        except IOError as e:
            print(f"Error saving file '{filepath}': {e}")
            return "" # Return empty string on error
