"""
This module contains the core logic for generating TLDR summaries.

It provides a class, TldrEngine, which is responsible for processing input files,
fetching additional context, performing Retrieval Augmented Generation (RAG)
using local embeddings, refining user queries, and generating summaries based on the
provided information. It manages interactions with a completion API and handles the
output of intermediate and final results.
"""

import re
import asyncio
from .openai import CompletionHandler


class TldrEngine(CompletionHandler):
    """
    Handles the core logic for generating TLDR summaries.

    This class is responsible for processing input files, fetching additional context,
    optionally performing Retrieval Augmented Generation (RAG) using local embeddings,
    refining user queries, and generating summaries based on the provided information.
    It manages interactions with a completion API and handles the output of
    intermediate and final results.

    Attributes:
        recursive (bool): Whether to search for input files recursively in directories.
        verbose (bool): If True, enables detailed logging.
        api_key (str, optional): The API key for the completion service.
        context_size (str): Specifies the context window size for the model (e.g., "small", "medium", "large").
        tag (str): A tag for the current run, used for naming output directories.
        output_directory (str): The path to the directory where intermediate and final files are saved.
        content (list[str]): A list of strings, where each string is the content of an input file.
        query (str): The user's query to be addressed by the summary.
        added_context (str): Formatted additional context to be used in prompts.
    """

    def __init__(
        self,
        verbose=True,
        api_key=None,
    ):
        # Initialize parent class chain
        super().__init__(api_key=api_key)

        # Additional attributes
        self.verbose = verbose
        self.random_seed = 42
        self.content = None
        self.query = None
        self.added_context = ""
        self.platform = "openai"

        # Initialize session
        self._generate_run_tag()
        self._create_output_path()
        self._start_logging()
        self._read_system_instructions()
        self._new_openai_client()

    async def load_all_content(
        self,
        input_files=None,
        context_files=None,
        context_size="medium",
        recursive=False,
    ):
        """Establish context for the TldrEngine"""

        # Read in files and contents
        if input_files is not None:
            self.content = self.fetch_content(user_files=input_files, label="reference")
        else:
            self.content = self.fetch_content(recursive=recursive, label="reference")

        if len(self.content.keys()) == 0:
            self.logger.error(f"No valid resources provided for reference documents.")
            return
        else:
            self.logger.info(f"Identified {len(self.content.keys())} reference documents.")

        # Create embeddings
        await self.encode_text_to_vector_store()

        # Fetch and reformat additional context if provided
        if context_files is not None:
            all_context = self.fetch_content(user_files=context_files, label="context")
            if len(all_context.keys()) == 0:
                self.logger.error(f"No valid resources provided for context documents.")
            else:
                self.logger.info(f"Identified {len(all_context.keys())} context documents.")
                await self.add_succinct_context(
                    self.join_text_objects(all_context, data_type="content"), 
                    context_size=context_size, 
                    label="context_files"
                )
        
    @staticmethod
    def strip_leading_bullet(text):
        """Removes a single leading bullet point and whitespace from a string."""
        text = text.lstrip()
        if text and text[0] in "*-+•":
            return text[1:].lstrip()
        return text

    async def initial_context_search(self, context_size="medium"):
        """Search for context in the vector store"""
        # Break up multiple queries
        queries = [self.strip_leading_bullet(query) for query in self.query.split("\n")]

        # Address each query separately
        self.logger.info("Searching for more context in provided documents and web...")
        new_context = []
        for query in queries:
            self.logger.info(f"Searching for context for query: {query}")
            new_context.append(f"Query: {query}")

            # Search embeddings
            current_search = await self.search_embedded_context(query=query)
            search_context = self.join_text_objects(list(current_search.keys()))
            new_context.append(f"Embedding search results: {search_context}")
            
            # Search web
            current_research = await self.perform_api_call(
                prompt=query,
                prompt_type="web_search", 
                context_size=context_size, 
                web_search=True
            )
            new_context.append(f"Web search results: {current_research['response']}")

        # Format context
        self.logger.info(f"Context search results: {len(new_context)}")
        self.context_search_results = new_context
        await self.add_succinct_context(
            self.join_text_objects(new_context), 
            context_size=context_size, 
            label="context_search"
        )

    async def finish_session(self):
        """Complete run with stats reporting"""
        result = self.save_response_text(self.added_context, label="full_context")
        self.logger.info(f"Response saved to: {result}")
        self.format_spending()
        self.generate_token_report()
        await self.client.close()

    async def add_succinct_context(
        self, new_context, context_size="medium", label="formatted_context"
    ):
        """
        Creates more concise, less repetative context message
        """
        # Format context references
        formatted_context = await self.perform_api_call(
            prompt=new_context,
            prompt_type="format_context",
            context_size=context_size,
            web_search=False,
        )
        self.added_context += formatted_context["response"]
        result = self.save_response_text(
            formatted_context["response"],
            label=label,
        )
        self.logger.info(f"Response saved to: {result}")

    def _lint_query(self, query) -> str:
        """
        Normalizes user query to ensure it's more well-formed & AI-friendly
        """
        # Normalize input to string
        if isinstance(query, list):
            processed_query = " ".join(map(str, query)).strip()
        else:
            processed_query = str(query).strip()

        # Ensure it ends with a suitable punctuation mark
        if processed_query[-1] not in ".!?":
            processed_query += "?"

        return processed_query

    async def refine_user_query(self, query, context_size="medium"):
        """
        Automatically improve user query for greater specificity
        """
        if query is None or query.strip() == "":
            self.query = ""
            return
        else:
            self.logger.info("Refining user query...")

        # Generate new query text
        refined_query = await self.perform_api_call(
            prompt=self._lint_query(query),
            prompt_type="refine_prompt",
            context_size=context_size,
            web_search=False,
        )

        # Handle output text
        self.query = refined_query["response"]
        self.logger.info(f"Refined query: {self.query}")
        result = self.save_response_text(self.query, label="refined_query")
        self.logger.info(f"Response saved to: {result}")
        self.logger.info(f"Finished refining user query")
        

    async def summarize_resources(self, context_size="medium"):
        """Generate component and synthesis summary text"""
        self.logger.info("Generating summaries for selected resources...")

        # Asynchronously summarize documents
        reference_summaries = await self.multi_completions(
            content_dict=self.content, context_size=context_size
        )

        # Annotate and save response strings
        all_summaries = []
        for i, summary in enumerate(reference_summaries):
            self.content[summary["source"]]["summary"] = summary["response"]
            all_summaries.append(f"Reference {i} Summary:\n{summary['response']}\n")
        self.all_summaries = self.join_text_objects(all_summaries)
        result = self.save_response_text(
            self.all_summaries, label="reference_summaries"
        )
        self.logger.info(f"Response saved to: {result}")
        self.logger.info(f"Finished summarizing resources")

    async def integrate_summaries(self, context_size="medium"):
        """Generate integrated executive summary text"""
        self.logger.info("Generating integrated summary text...")

        # Generate executive summary
        executive_summary = await self.perform_api_call(
            prompt=self.all_summaries,
            prompt_type="executive_summary",
            context_size=context_size,
            web_search=False,
        )
        self.executive_summary = executive_summary["response"]

        # Handle output
        result = self.save_response_text(
            executive_summary["response"], label="executive_summary"
        )
        self.logger.info(f"Response saved to: {result}")

        self.logger.info(f"Finished integrating summaries")

    async def polish_response(self, tone="stylized", context_size="medium"):
        """Refine final response text"""
        self.logger.info("Finalizing response text...")

        # Select tone
        tone_instructions = (
            "formal_polishing" if tone == "formal" else "stylized_polishing"
        )

        # Polish summary
        self.polished_summary = await self.perform_api_call(
            prompt=self.executive_summary,
            prompt_type=tone_instructions,
            context_size=context_size,
            web_search=False,
        )
        self.polished_summary = self.polished_summary["response"]
        result = self.save_response_text(
            self.polished_summary,
            label="polished_summary",
        )
        self.logger.info(f"Response saved to: {result}")

        self.logger.info(f"Finished polishing response")

    async def save_to_pdf(self, smart_title=True, polished=True):
        """Saves polished summary string to formatted PDF document."""
        self.logger.info("Saving final summary to PDF...")
        summary_text = (
            self.polished_summary if polished is True else self.executive_summary
        )

        # Generate title
        if smart_title is True:
            doc_title = await self.perform_api_call(
                prompt=summary_text,
                prompt_type="title_instructions",
                context_size="low",
                web_search=False,
            )

        # Save to PDF
        pdf_path = self.generate_tldr_pdf(
            summary_text=summary_text,
            doc_title=doc_title["response"],
        )
        self.logger.info(f"Final summary saved to {pdf_path}")

    async def determine_prompt_type(self, text, filename=None, web_search=False):
        """Determine type of submission"""
        if web_search is False:
            self.logger.info(f"Determining file type of {filename}...")
            source_type = await self.perform_api_call(
                prompt=text,
                prompt_type="file_type",
                web_search=web_search,
            )
            self.logger.info(f"File type determined: {source_type['response']}")
            file_type = re.sub(r"[^a-zA-Z]", "", source_type["response"].lower())
            return file_type
        else:
            return "web_search"

    async def multi_completions(
        self,
        content_dict,
        context_size="medium",
        web_search=False,
        **kwargs,
    ):
        """
        Runs multiple completion calls concurrently.
        """
        # Parse all of the content found
        coroutine_tasks = []
        for source in content_dict.keys():

            # Determine type of submission
            content = content_dict[source]["content"]
            prompt_type = await self.determine_prompt_type(content, source, web_search)

            # Generate summary task
            task = self.perform_api_call(
                prompt=content,
                prompt_type=prompt_type,
                web_search=web_search,
                context_size=context_size,
                source=source,
                **kwargs,
            )
            coroutine_tasks.append(task)

        return await asyncio.gather(*coroutine_tasks, return_exceptions=False)

    async def apply_research(self, context_size="medium"):
        """
        Identify knowledge gaps and use web search to fill them.
        """
        self.logger.info("Identifying knowledge gaps...")
        research_questions = await self.perform_api_call(
            prompt=self.all_summaries,
            prompt_type="background_gaps",
            context_size=context_size,
            web_search=False,
        )

        # Reassemble research questions into content dictionary
        research_questions = research_questions["response"].split("\n")
        research_question_dict = {}
        for i, question in enumerate(research_questions):
            question = self.strip_leading_bullet(question)
            research_question_dict[f"Question_{i}"] = {"content": question}

        # Search web for to fill gaps
        self.logger.info("Applying web research agent to knowledge gaps...")
        self.research_results = await self.multi_completions(
            content_dict=research_question_dict,
            context_size=context_size,
            web_search=True,
        )

        # Also search for additional context missed in references
        self.logger.info("Searching for additional context in reference docs...")
        research_context = []
        for q_ans_pair in self.research_results:
            self.logger.info(f"Potential knowledge gap: {q_ans_pair['prompt']}")
            search_context = self.search_embedded_context(
                query=q_ans_pair["prompt"], num_results=1
            )
            search_context = self.join_text_objects(list(search_context.keys()))
            current_context = f'Question:\n{q_ans_pair["prompt"]}\n'
            current_context += f'Answer:\n{q_ans_pair["response"]}\n'
            current_context += f"Local RAG Context:\n{search_context}\n"
            research_context.append(current_context)

        # Assemble full research context
        divider = "\n#--------------------------------------------------------#\n"
        research_str = self.join_text_objects(research_context, divider)
        self.research_context = research_str

        # Update added context with new research
        await self.add_succinct_context(research_str, label="research_context")
        result = self.save_response_text(research_str, label="research_results")
        self.logger.info(f"Response saved to: {result}")

        self.logger.info(f"Finished research knowledge gaps")
