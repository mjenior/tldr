"""Utility module for TLDR command-line argument parsing.

This module provides utility functions for argument parsing in the TLDR application.

1. parse_tldr_arguments(): Parses arguments for the main TLDR summarization functionality.
   - Handles file input, query refinement, context files, recursive search, web search,
     response tone, context size, chunk splitting, and verbosity settings.

2. parse_eval_arguments(): Parses arguments for evaluating summaries against target content.
   - Manages content paths, summary paths, evaluation model selection,
     iteration count, and verbosity settings.

The module uses argparse for argument parsing and provides clear, user-friendly help messages
for all available options.
"""

import argparse


def parse_tldr_arguments():
    """Parse command-line arguments for the TLDR application.

    This function sets up and parses the command-line arguments that control
    the behavior of the application. It defines options for specifying the
    user query, input files, context files, and various other settings.

    Returns:
        argparse.Namespace: An object containing the parsed command-line
            arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="TLDR: Summarize text files based on user arguments.",
    )
    # Define arguments
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Optional user query",
    )
    parser.add_argument(
        "--input_files",
        default=None,
        nargs="+",
        help="Optional: Input files to summarize (Default is scan for text files in working directory)",
    )
    parser.add_argument(
        "--refine_query",
        type=bool,
        default=True,
        help="Automatically refine and improve the user query",
    )
    parser.add_argument(
        "--context_files",
        default=None,
        nargs="+",
        help="Optional: Additional context files to script include in the summary",
    )
    parser.add_argument(
        "--recursive_search",
        action="store_true",
        help="Recursively search input directories",
    )
    parser.add_argument(
        "--web_search",
        type=bool,
        default=True,
        help="Additional research agent to find and fill knowledge gaps",
    )
    parser.add_argument(
        "--context_size",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Modifier for scale of maximum output tokens and context window size for research agent web search",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Testing mode",
    )
    parser.add_argument(
        "--polish",
        type=bool,
        default=True,
        help="Polish final response",
    )
    parser.add_argument(
        "--tone",
        choices=["stylized", "formal"],
        default="stylized",
        help="Tone of polishing to apply to the final response. Options: 'stylized' (default) or 'formal'.",
    )
    parser.add_argument(
        "--pdf",
        type=bool,
        default=True,
        help="Save final response to PDF",
    )
    parser.add_argument(
        "--verbose", type=bool, default=True, help="Verbose stdout reporting"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI",
    )
    parser.add_argument(
        "--injection_screen",
        type=bool,
        default=True,
        help="Enable prompt injection testing",
    )
    return parser.parse_args()


def parse_eval_arguments():
    """Parse command-line arguments for the evaluation functionality.

    This function sets up and parses the command-line arguments that control
    the behavior of the evaluation functionality. It defines options for
    specifying the content to evaluate, the summary to evaluate, the model to
    use for evaluation, and the number of iterations to run.

    Returns:
        argparse.Namespace: An object containing the parsed command-line
            arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a summary against the target content.",
    )

    # Define arguments
    parser.add_argument(
        "--content",
        default=None,
        required=True,
        help="Content to evaluate summary against",
    )
    parser.add_argument(
        "--summary",
        default=None,
        required=True,
        help="Summary to evaluate",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of iterations to run",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Verbose stdout reporting",
    )

    return parser.parse_args()
