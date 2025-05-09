#!/usr/bin/env python3

import argparse
from .core import TldrClass
from .utils import parse_user_arguments, save_response_text


if __name__ == "__main__":
    args = parse_user_arguments()

    # Read in content and extend user query
    tldr = TldrClass(query=args.query, search_directory=args.input_directory)

    # Summarize documents
    summaries = await tldr.async_completion(
        prompts=prompts_list,
        instructions=system_instructions)
    summary = "\n\n".join(summaries)
    save_response_text(summary, label='summaries', output_dir=args.output_directory)

    # TODO: Generate synthesis text - address query if provided
    synthesis = await tldr.async_multi_completion(
        prompts=summaries,
        instructions=system_instructions)
    
    # TODO: Use research agents to fill gaps
    #if args.research == True:
        # Identify gaps in understanding

        # Search web for to fill gaps

        # Integrate findings in synthesis text

    # Save synthesis text
    save_response_text(synthesis, label='synthesis', output_dir=args.output_directory)

