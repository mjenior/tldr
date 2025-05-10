#!/usr/bin/env python3

import argparse
from .core import TldrClass
from .utils import parse_user_arguments, save_response_text


if __name__ == "__main__":
    # Read in command line arguments
    args = parse_user_arguments()

    # Read in content and extend user query
    tldr = TldrClass(
        query=args.query, 
        search_directory=args.input_directory, 
        verbose=args.verbose)

    # Summarize documents
    summary = tldr.summarize_resources()

    # Use research agent to fill gaps
    if args.research == True:
        summary = self.apply_research(summary)

    # Rewrite for respone type and tone
    final = self.polish_response(summary, query, args.output_type)
