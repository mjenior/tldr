import re
import asyncio

from .completion import CompletionHandler
from .utils import encode_text
from .dartboard import DartboardRetriever


class TldrEngine(CompletionHandler):
    """
    Handles the core logic for generating TLDR summaries.

    This class is responsible for processing input files, fetching additional context,
    optionally performing Retrieval Augmented Generation (RAG) using local embeddings,
    refining user queries, and generating summaries based on the provided information.
    It manages interactions with a completion API and handles the output of
    intermediate and final results.

    Attributes:
        input_files (list[str], optional): A list of file paths to be summarized.
        context_files (list[str], optional): A list of file paths to be used as additional context.
        recursive_search (bool): Whether to search for input files recursively in directories.
        verbose (bool): If True, enables detailed logging.
        api_key (str, optional): The API key for the completion service.
        local_rag (bool): If True, enables local Retrieval Augmented Generation.
        context_size (str): Specifies the context window size for the model (e.g., "small", "medium", "large").
        tag (str): A tag for the current run, used for naming output directories.
        output_directory (str): The path to the directory where intermediate and final files are saved.
        content (list[str]): A list of strings, where each string is the content of an input file.
        raw_context (str, optional): Combined content of all context files.
        dartboard (DartboardRetriever, optional): Retriever instance for local RAG if `local_rag` is True.
        query (str): The user's query to be addressed by the summary.
        added_context (str): Formatted additional context to be used in prompts.
    """

    def __init__(
        self,
        input_files=None,
        query="",
        refine_query=True,
        context_files=None,
        recursive_search=False,
        verbose=True,
        api_key=None,
        context_size="medium",
        tone="default",
        testing=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.testing = testing
        self.context_size = context_size
        self.recursive_search = recursive_search
        self.query = query
        self.refine_query = refine_query
        self.added_context = ""
        self.tone = tone
        self.research = True

        # Set up logger
        self.setup_logging()

        # Establish intermediate file output directory
        self._create_output_path()
        self.logger.info(
            f"Intermediate files being written to: {self.output_directory}"
        )

        # Read in system instructions
        self.instructions = self.read_system_instructions()

        # Read in files and contents
        self.logger.info("Searching for input files...")
        if input_files is not None:
            self.content = self.fetch_content(user_files=input_files)
        else:
            self.content = self.fetch_content(recursive=self.recursive_search)
        # Check if no resources were found
        if len(self.content) == 0:
            self.logger.error(
                "No resources found in current search directory. Exiting."
            )
            sys.exit(1)
        else:
            self.logger.info(
                f"Identified {len(self.content)} reference documents for summarization."
            )

        # Fetch additional context if provided
        if context_files is not None:
            all_context = self.fetch_content(user_files=context_files)
            self.logger.info(
                f"Also identified {len(all_context)} reference documents for added context."
            )
            self.raw_context = "\n".join(all_context)
        else:
            self.raw_context = None

        # Create local embeddings
        if self.query is not None or self.research is True:
            self.logger.info(f"Generating embeddings for reference documents...")
            self.dartboard = DartboardRetriever(
                chunks=encode_text(self.content),
                default_num_results=5,
                default_oversampling_factor=3,
                default_div_weight=1.0,
                default_rel_weight=1.0,
                default_sigma=0.1,
            )

    async def format_context_references(self):
        """Creates more concise, less repetative context message."""

        added_context = asyncio.run(
            self.single_completion(
                message="\n".join(self.raw_context), prompt_type="format_context"
            )
        )
        result = self.save_response_text(
            added_context,
            label="added_context",
            output_dir=self.output_directory,
        )
        self.logger.info(result)
        self.added_context += f"\nBelow is additional context for reference during response generation:\n{added_context}"

    @staticmethod
    def _lint_query(current_query: str | list) -> str:
        """
        Normalizes a user query to ensure it's a well-formed, AI-friendly string.
        """
        # Normalize input to string
        if isinstance(current_query, list):
            processed_query = " ".join(map(str, current_query))
        else:
            processed_query = str(current_query)

        # Strip leading/trailing whitespace and normalize internal whitespace
        processed_query = re.sub(r"\s+", " ", processed_query.strip())

        # Capitalize first letter if not already
        processed_query = processed_query[0].upper() + processed_query[1:]

        # Ensure it ends with a suitable punctuation mark
        if processed_query[-1] not in ".!?":
            processed_query += "?"

        return processed_query

    async def refine_user_query(self):
        """Attempt to automatically improve user query for greater specificity"""

        # Check text formatting
        self.query = self._lint_query(self.query)

        # Generate new query text
        refined_query = await self.single_completion(
            message=self.query, prompt_type="refine_prompt"
        )
        self.logger.info(refined_query)

        # Handle output text
        self.query = f"Ensure that addressing the following user query is the key consideration in you response:\n{refined_query}\n"
        result = self.save_response_text(
            refined_query,
            label="refined_query",
            output_dir=self.output_directory,
        )
        self.logger.info(result)

    async def summarize_resources(self):
        """Generate component and synthesis summary text"""

        # Asynchronously summarize documents
        reference_summaries = await self.multi_completions(self.content)

        # Annotate and save response strings
        self.reference_summaries = []
        for i, summary in enumerate(reference_summaries):
            self.reference_summaries.append(
                f"Reference {i} Summary:\n{summary["response"]}\n"
            )
            result = self.save_response_text(
                summary["response"],
                label="summary",
                output_dir=self.output_directory,
                idx=i,
            )
            self.logger.info(result)

    async def integrate_summaries(self):
        """Generate integrated executive summaries combine all current summary text"""
        self.logger.info("Generating integrated summary text...")

        # Join all summaries for one large submission
        all_summary_text = "\n".join(self.reference_summaries)
        executive_summary = await self.single_completion(
            message=all_summary_text, prompt_type="executive_summary"
        )
        self.executive_summary = executive_summary["response"]

        # Handle output
        result = self.save_response_text(
            executive_summary["response"],
            label="synthesis",
            output_dir=self.output_directory,
        )
        self.logger.info(result)

    async def apply_research(self):
        """Identify knowledge gaps and use web search to fill them. Integrating new info into summary."""
        self.logger.info("Applying research agent to knowledge gaps...")

        # Identify gaps in understanding
        research_questions = await self.single_completion(
            message=self.executive_summary, prompt_type="background_gaps"
        )
        research_questions = research_questions["response"].split("\n")

        # Search web for to fill gaps
        self.research_results = await self.multi_completions(
            research_questions, prompt_type="web_search"
        )

        # Handle output
        result = self.save_response_text(
            "\n".join(
                [
                    f'Question:{q_ans_pair["prompt"]}:\nAnswer:{q_ans_pair["response"]}'
                    for q_ans_pair in self.research_results
                ]
            ),
            label="research",
            output_dir=self.output_directory,
        )
        self.logger.info(result)

    async def polish_response(self):
        """Refine final response text"""
        self.logger.info("Finalizing response text...")

        # Select tone
        tone_instructions = (
            "polishing" if self.tone == "default" else "modified_polishing"
        )

        # Run completion and save final response text
        research_answers = "\n".join([x["response"] for x in self.research_results])
        self.polished_summary = await self.single_completion(
            message=self.executive_summary + research_answers + self.added_context,
            prompt_type=tone_instructions,
        )
        polished_title = await self.single_completion(
            message=self.polished_summary["response"], prompt_type="title_instructions"
        )

        # Save formatted PDF
        pdf_path = generate_tldr_pdf(
            self.polished_summary["response"], polished_title["response"]
        )

        return f"Final summary saved to {pdf_path}\n"
