#!/usr/bin/env python3

import sys
import logging
import argparse
import asyncio

from .core import TldrClass
from .logging import setup_logging


def parse_user_arguments():
    """Parses command-line arguments"""
    print("")
    parser = argparse.ArgumentParser(
        description="TLDR: Summarize text files based on user arguments."
    )

    # Define arguments
    parser.add_argument("query", nargs="?", default="", help="Optional user query")
    parser.add_argument(
        "-i",
        "--input_directory",
        default=".",
        help="Directory to scan for text files (Default is working directory)",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        default=".",
        help="Directory for output files (Default is working directory)",
    )
    parser.add_argument(
        "-q",
        "--refine_query",
        type=bool,
        default=True,
        help="Automatically refine and improve the user query",
    )
    parser.add_argument(
        "-c",
        "--context_directory",
        default=None,
        help="Directory for additional context documents",
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
        "-v", "--verbose", type=bool, default=False, help="Verbose stdout reporting"
    )

    parser.add_argument(
        "--testing",
        type=bool,
        default=False,
        help="Uses only gpt-4o-mini for cheaper+faster testing runs.",
    )

    return parser.parse_args()


async def main():
    """Main entry point for the tldr command"""

    # Read in command line arguments
    args = parse_user_arguments()

    # Set up logger
    setup_logging(args.verbose)
    main_logger = logging.getLogger()

    # Read in content and extend user query
    tldr = TldrClass(
        search_directory=args.input_directory,
        output_directory=args.output_directory,
        context_directory=args.context_directory,
        recursive_search=args.recursive_search,
        context_size=args.context_size,
        verbose=args.verbose,
        testing=args.testing,
    )

    # Check if no resources were found
    if tldr.sources == 0:
        main_logger.error("No resources found in current search directory. Exiting.")
        sys.exit(1)

    # Report something if verbose is False
    if args.verbose == False:
        print("TLDR is working, this may take a minute...")

    # Extend user query
    if len(args.query) > 0 and args.refine_query == True:
        main_logger.info("Refining user query...")
        await tldr.refine_user_query(args.query)

    # Condense any context docs provided
    if tldr.raw_context is not None:
        main_logger.info("Refining background context...")
        await tldr.format_context_references()

    # Summarize documents
    main_logger.info("Generating summaries for selected resources...")
    await tldr.summarize_resources()

    # Synthesize content
    if len(tldr.all_summaries) >= 2:
        main_logger.info("Generating integrated summary text...")
        await tldr.integrate_summaries()
    else:
        tldr.content_synthesis = tldr.all_summaries[0]

    # Use research agent to fill gaps
    if args.web_search:
        main_logger.info("Applying research agent to knowledge gaps...")
        await tldr.apply_research()

    # Rewrite for response type and tone
    main_logger.info("Finalizing response text...")
    polishing_result = await tldr.polish_response(args.tone)

    # Complete run
    tldr.format_spending()
    main_logger.info(f"Session Cost: {tldr.total_spend}")
    tldr.generate_token_report()
    main_logger.info(f"Session Token Usage: {tldr.token_report}")
    print(polishing_result)


def cli_main():
    """Command line utility function call"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
