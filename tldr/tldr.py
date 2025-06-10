#!/usr/bin/env python3

import asyncio

from .core import TldrEngine
from .utils import parse_user_arguments


async def main():
    """Main entry point for the tldr command"""

    # Read in content and extend user query
    tldr = TldrEngine(**parse_user_arguments())

    # Report something if verbose is False
    if tldr.verbose is False:
        print("TLDR is working, this may take a minute...")

    # Create local embeddings
    if tldr.query is not None or tldr.research is True:
        tldr.encode_text_to_vector_store()

    # Extend user query
    await tldr.refine_user_query()

    # Query local embedded text vectors
    if tldr.query != "":
        await tldr.search_embedded_context()

    # Condense any context docs provided
    if tldr.raw_context is not None:
        await tldr.format_context_references()

    # Summarize documents
    await tldr.summarize_resources()

    # Synthesize content
    await tldr.integrate_summaries()

    # Use research agent to fill gaps
    if tldr.web_search:
        await tldr.apply_research()

    # Rewrite for response type and tone
    await tldr.polish_response()
    tldr.logger.info(tldr.polished_summary)

    # Complete run with stats reporting
    tldr.format_spending()
    tldr.generate_token_report()


def cli_main():
    """Command-line entry point to run the tldr script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
