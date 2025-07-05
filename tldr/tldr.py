#!/usr/bin/env python3
"""TL;DR - A command-line tool for generating concise, AI-powered summaries of documents and resources.

This module provides the main entry point for the TL;DR command-line tool, which processes documents,
web resources, or other content to generate clear and concise summaries. It leverages AI to understand
context, extract key information, and present it in a digestible format.

Key Features:
- Processes various input sources (local files, URLs, etc.)
- Uses AI to understand and summarize content
- Supports different output formats and styles
- Maintains context across multiple documents
- Can perform research to fill information gaps
- Saves results in multiple formats including PDF

Usage:
    tldr [OPTIONS] [INPUT_SOURCE]

Example:
    tldr --input_files document.txt --context_files context.md --query "What is the purpose of this document?" --tone formal

Note:
    This module requires an active internet connection and appropriate API keys for AI services.
"""

import asyncio

from .core import TldrEngine
from .args import parse_tldr_arguments


async def pipeline():
    """Main entry point for the tldr command"""

    # Initialize TldrEngine
    args = parse_tldr_arguments()
    tldr = TldrEngine(
        verbose=args.verbose,
        api_key=args.api_key,
        injection_screen=args.injection_screen,
    )

    # Optional: Refine user query
    if args.refine_query is True and args.testing is False:
        await tldr.refine_user_query(query=args.query, context_size=args.context_size)

    # Establish context
    await tldr.load_all_content(
        input_files=args.input_files,
        context_files=args.context_files,
        context_size=args.context_size,
        recursive=args.recursive_search,
    )

    # Establish initial context
    if args.query is not None and args.testing is False:
        await tldr.initial_context_search(context_size=args.context_size)

    # Summarize documents
    await tldr.summarize_resources(context_size=args.context_size)

    # Optional: Use research agent to fill gaps
    if args.research is True and args.testing is False:
        await tldr.apply_research(context_size=args.context_size)

    # Synthesize content
    await tldr.integrate_summaries(context_size=args.context_size)

    # Optional: Rewrite for response type and tone
    if args.polish is True and args.testing is False:
        await tldr.polish_response(tone=args.tone, context_size=args.context_size)

    # Optional: Save final context
    if args.pdf is True and args.testing is False:
        await tldr.save_to_pdf(polished=args.polish)

    # End session
    await tldr.finish_session()


def run_tldr():
    """Command-line entry point to run the tldr script."""
    asyncio.run(pipeline())


if __name__ == "__main__":
    run_tldr()
