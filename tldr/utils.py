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
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="TLDR: Summarize text files based on user arguments."
    )

    # Define arguments
    parser.add_argument("query", nargs="?", default=None, help="Optional user query")
    parser.add_argument(
        "-i",
        "--input_files",
        default=None,
        nargs="+",
        help="Optional: Input files to summarize (Default is scan for text files in working directory)",
    )
    parser.add_argument(
        "-f",
        "--refine_query",
        type=bool,
        default=True,
        help="Automatically refine and improve the user query"
    )
    parser.add_argument(
        "-c",
        "--context_files",
        default=None,
        nargs="+",
        help="Optional: Additional context files to include in the summary"
    )
    parser.add_argument(
        "-r",
        "--recursive_search",
        type=bool,
        default=False,
        help="Recursively search input directories",
    )
    parser.add_argument(
        "-w",
        "--web_search",
        type=bool,
        default=True,
        help="Additional research agent to find and fill knowledge gaps",
    )
    parser.add_argument(
        "-t",
        "--tone",
        choices=["default", "modified"],
        default="default",
        help="Final executive summary response tone",
    )
    parser.add_argument(
        "-s",
        "--context_size",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Modifier for scale of maximum output tokens and context window size for research agent web search",
    )
    parser.add_argument(
        "-sc",
        "--split_chunks",
        type=bool,
        default=False,
        help="Split summaries into chunks before shuffling",
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=True, help="Verbose stdout reporting"
    )

    return vars(parser.parse_args())


def parse_eval_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate a summary against the target content."
    )

    # Define arguments
    parser.add_argument(
        "-c",
        "--content",
        default=None,
        required=True,
        help="Content to evaluate summary against",
    )
    parser.add_argument(
        "-s",
        "--summary",
        default=None,
        required=True,
        help="Summary to evaluate",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=True,
        help="Verbose stdout reporting",
    )

    return vars(parser.parse_args())
