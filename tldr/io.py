import os
import re
import sys
import yaml
from datetime import datetime

from docx import Document
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


class FileHandler:

    def create_timestamp(self):
        """Generate a timestamp string (e.g., 20231027_103000)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_tag = f"tldr.{timestamp}"

    def _create_output_path(self):
        """Set up where to write intermediate files"""
        self.output_directory = f"{self.run_tag}.files"
        os.makedirs(self.output_directory, exist_ok=True)
        self.logger.info(
            f"Intermediate files being written to: {self.output_directory}"
        )

    def fetch_content(
        self,
        label: str = "input",
        user_files: list = None,
        search_dir: str = ".",
        recursive: bool = False,
    ):
        """
        Find files and read in their contents.
        Returns a list of content strings.
        """
        self.logger.info(f"Searching for {label} documents...")
        files = self._find_readable_files(
            infiles=user_files, directory=search_dir, recursive=recursive
        )

        content = []
        for ext in files.keys():
            content += self.read_file_content(files[ext], ext)

        # Check if no resources were found
        if len(content) == 0:
            self.logger.error(
                "No resources found in current search directory. Exiting."
            )
            sys.exit(1)
        else:
            self.logger.info(f"Identified {len(content)} {label} documents.")
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
        readable_files_by_type = {
            "pdf": [],
            "docx": [],
            "html": [],
            "txt": [],
            "md": [],
        }

        if infiles is not None:
            for file_path_item in infiles:
                ext = os.path.splitext(file_path_item)[1].lower()
                self._update_file_dictionary(
                    readable_files_by_type, file_path_item, ext
                )
        else:
            if not recursive:
                for filename in os.listdir(directory):
                    if ".tldr." in filename:
                        continue
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        ext = os.path.splitext(filename)[1].lower()
                        self._update_file_dictionary(
                            readable_files_by_type, filepath, ext
                        )
            else:
                for root, _, files_in_dir in os.walk(directory):
                    for filename in files_in_dir:
                        if ".tldr." in filename:
                            continue
                        filepath = os.path.join(root, filename)
                        ext = os.path.splitext(filename)[1].lower()
                        self._update_file_dictionary(
                            readable_files_by_type, filepath, ext
                        )
        return {k: v for k, v in readable_files_by_type.items() if v}

    def _update_file_dictionary(self, file_dict, file_path, file_ext):
        """Add file path to dictionary if readable and of a supported type."""
        try:
            if file_ext == ".pdf":
                reader = PdfReader(file_path)
                if reader.pages and any(
                    page.extract_text() for page in reader.pages if page.extract_text()
                ):
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
                current_ext = os.path.splitext(filepath)[1].lower().lstrip(".")

            try:
                if current_ext == "pdf":
                    reader = PdfReader(filepath)
                    text = ""
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

            except Exception as e:  # pylint: disable=broad-except
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
        except Exception as e:  # pylint: disable=broad-except
            print(
                f"An unexpected error occurred while reading '{instructions_path}': {e}"
            )
        return instructions

    def save_response_text(
        self,
        out_data: str,
        label: str = "response",
        output_dir: str = ".",
        idx: int = 1,
        chunk_size: int = 1024 * 1024,
        errors: str = "strict",
    ) -> str:
        """
        Saves a string variable to a text file with a dynamic filename.
        This method now incorporates the logic of the former _save_summary_txt function.
        Returns the full path to the saved file, or an empty string on error.
        """
        if label == "summary":
            filename = f"{label}.{idx}.{self.run_tag}.txt"
        else:
            filename = f"{label}.{self.run_tag}.txt"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8", errors=errors) as f:
                for i in range(0, len(out_data), chunk_size):
                    f.write(out_data[i : i + chunk_size])
            return filepath
        except IOError as e:
            return f"Error saving file '{filepath}': {e}"

    def generate_tldr_pdf(self, summary_text, doc_title):
        """Saves polished summary string to formatted PDF document."""

        # Create file path with linted name
        file_path = self._create_filename(doc_title)

        # Format content
        styles = getSampleStyleSheet()
        h1_style = styles["h1"]
        h1_style.alignment = 1
        body = [
            Paragraph(doc_title.replace('"', ""), h1_style),
            Spacer(1, 0.15 * inch),
        ]
        body += self._interpret_markdown(summary_text)

        # Create document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        try:
            doc.build(body)
        except Exception as e:
            print(f"An unexpected error occurred while saving {file_path}: {e}")
            raise

        return file_path

    @staticmethod
    def _interpret_markdown(text: str) -> list:
        """
        Converts custom markdown-like syntax to ReportLab-friendly HTML.
        - # Header 1      → <font size=18>
        - ## Header 2    → <font size=16>
        - ### Header 3  → <font size=14>
        - #### Header 4   → <font size=12>
        - *bold*          → <b>
        - ~italic~    → <i>
        Returns a list of Paragraph and Spacer elements.
        """
        styles = getSampleStyleSheet()
        normal = styles["Normal"]
        story = []

        # Header sizes mapping
        header_sizes = {1: 16, 2: 14, 3: 12, 4: 11}

        lines = text.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.1 * inch))
                continue
            elif line.startswith("# "):
                continue
            elif line in ["###", "---"]:
                continue

            # Convert italic/bold (model uses *, shrugs) and subscripts
            line = re.sub(r"\*(.*?)\*", r"<b>\1</b>", line)
            line = re.sub(r"_(.*?)_", r"<sub>\1</sub>", line)

            # Header detection: one or more '#' followed by a space
            header_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2).strip()
                font_size = header_sizes.get(level, 11)  # default to 11 for > 4 #
                html_line = f'<b><font size="{font_size}">{content}</font></b>'
                story.append(Paragraph(html_line, normal))
                story.append(Spacer(1, 0.05 * inch))
            elif line.startswith("- "):
                bullet_content = line[2:].strip()
                html_line = f"• {bullet_content}"
                story.append(Paragraph(html_line, normal))
            else:
                html_line = line
                story.append(Paragraph(html_line, normal))

        return story

    @staticmethod
    def _create_filename(title: str, max_length: int = 50) -> str:
        """
        Accepts a string that is a document title, removes uninformative words,
        and reformats the string to be used as a file name.
        """
        # Convert to lowercase
        filename = title.lower()

        # Define uninformative words (can be extended)
        uninformative_words = [
            "a",
            "an",
            "the",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "from",
            "by",
            "as",
            "that",
            "which",
            "this",
            "these",
            "those",
            "it",
            "its",
            "about",
            "through",
            "beyond",
            "up",
            "down",
            "out",
            "into",
            "over",
            "under",
            "from",
            "around",
            "about",
            "via",
            "re",
            "regarding",
            "concerning",
            "document",
            "report",
            "summary",
            "efficient",
            "technical",
            "overview",
            "introduction",
            "advancements",
            "analysis",
            "study",
            "research",
            "paper",
            "article",
            "draft",
            "final",
            "version",
            "update",
            "notes",
            "memo",
            "brief",
            "presentation",
            "review",
            "whitepaper",
            "guide",
            "manual",
            "spec",
            "specification",
            "appendix",
            "chapter",
            "section",
            "part",
            "volume",
            "issue",
            "release",
            "plan",
            "project",
            "initiative",
            "program",
            "system",
            "process",
            "procedure",
            "framework",
            "methodology",
            "approach",
            "solution",
            "strategy",
            "tbd",
            "for review",
            "confidential",
        ]

        # Remove uninformative words
        for word in uninformative_words:
            filename = re.sub(r"\b" + re.escape(word) + r"\b", "", filename)

        # Replace non-alphanumeric characters (except spaces and underscores)
        filename = re.sub(r"[^a-z0-9\s_]", " ", filename)

        # Replace white space with underscore
        filename = "_".join(filename.split()).strip("_")

        # Truncate if too long
        if len(filename) > max_length:
            filename = filename[:max_length]
            # Try to avoid cutting off in the middle of a word if possible
            last_underscore = filename.rfind("_")
            if last_underscore != -1 and last_underscore > max_length - 20:
                filename = filename[:last_underscore]

        # Assemble final name
        return filename.strip("_") + ".tldr.pdf"
