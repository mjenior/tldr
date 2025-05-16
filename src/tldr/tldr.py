#!/usr/bin/env python3

import sys
import argparse
import asyncio

from tldr.core import TldrClass


def parse_user_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description="TLDR: Summarize text files based on user arguments.")

    # Define arguments
    parser.add_argument("query", nargs='?', default="",
        help="Optional user query")
    parser.add_argument("-i", "--input_directory", type=str, default=".", 
        help="Directory to scan for text files (Default is working directory)")
    parser.add_argument("-o", "--output_directory", default=".", 
        help="Directory for output files (Default is working directory)")
    parser.add_argument("-r", "--refine_query", type=bool, default=True, 
        help="Verbose stdout reporting")
    parser.add_argument("-v", "--verbose", type=bool, default=True, 
        help="Verbose stdout reporting")
    parser.add_argument("-x", "--research", type=bool, default=True, 
        help="Additional research agent.")
    parser.add_argument("-t", "--output_type", choices=["default", "modified"], default="default",
        help="Response tone.")
    parser.add_argument("-g", "--glyphs", type=bool, default=False,
        help="Utilize associative glyphs during executive summary.")

    return parser.parse_args()


if __name__ == "__main__":
    # Read in command line arguments
    args = parse_user_arguments()

    # Read in content and extend user query
    tldr = TldrClass(user_query=args.query, search_directory=args.input_directory, verbose=args.verbose, glyphs=args.glyphs)
    if tldr.sources == 0:
        sys.exit(1)

    # Extend user query
    if args.refine_query == True and len(tldr.user_query) > 0: tldr.refine_user_query()

    # Summarize documents
    tldr.all_summaries = asyncio.run(tldr.summarize_resources())

    # Synthesize content
    tldr.integrate_summaries()

    # Use research agent to fill gaps
    if args.research == True: tldr.apply_research()
    
    # Rewrite for respone type and tone
    tldr.polish_response(args.output_type)
