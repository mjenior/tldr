#!/usr/bin/env python3

import sys
import argparse
import asyncio
from tldr.core import TldrClass


def parse_user_arguments():
    """Parses command-line arguments"""
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
        "-r",
        "--refine_query",
        type=bool,
        default=False,
        help="Automatically refine and improve the user query",
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=True, help="Verbose stdout reporting"
    )
    parser.add_argument(
        "-s",
        "--research",
        type=bool,
        default=True,
        help="Additional research agent to find and fill knowledge gaps",
    )
    parser.add_argument(
        "-c",
        "--context_size",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Context window size for research agent web search",
    )
    parser.add_argument(
        "-n",
        "--tone",
        choices=["default", "modified"],
        default="default",
        help="Final executive summary response tone",
    )
    parser.add_argument(
        "-g",
        "--glyphs",
        type=bool,
        default=False,
        help="Utilize associative glyphs during executive summary synthesis",
    )
    parser.add_argument(
        "-t",
        "--token_scale",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Modifier for scale of maximum output tokens window",
    )

    return parser.parse_args()


def main():
    """Main entry point for the tldr command"""
    # Read in command line arguments
    args = parse_user_arguments()

    # Read in content and extend user query
    tldr = TldrClass(
        search_directory=args.input_directory,
        output_directory=args.output_directory,
        verbose=args.verbose,
        token_scale=args.token_scale,
    )

    # Check if no resources were found
    if tldr.sources == 0:
        sys.exit(1)

    # Extend user query
    if args.refine_query == True and args.query is not None:
        tldr.user_query = tldr.refine_user_query(args.query)
    else:
        tldr.user_query = ""

    # Summarize documents
    tldr.all_summaries = asyncio.run(tldr.summarize_resources())

    # Synthesize content
    if len(tldr.all_summaries) >= 2:
        tldr.integrate_summaries(args.glyphs)
    else:
        tldr.content_synthesis = self.all_summaries[0]

    # Use research agent to fill gaps
    if args.research:
        tldr.apply_research(context_size=args.context_size)

    # Rewrite for response type and tone
    tldr.polish_response(args.tone)


if __name__ == "__main__":
    main()
