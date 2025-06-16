#!/usr/bin/env python3

import asyncio

from .core import TldrEngine
from .args import parse_tldr_arguments


async def main():
    """Main entry point for the tldr command"""

    # Read in content and extend user query
    tldr = TldrEngine(**parse_tldr_arguments())

    # Extend user query
    await tldr.refine_user_query()

    # Query local embedded text vectors
    if tldr.query != "":
        context_search = await tldr.search_embedded_context(query=tldr.query)
        await tldr.format_context(context_search, label="context_search")

    # Summarize documents
    await tldr.summarize_resources()

    # Use research agent to fill gaps
    if tldr.web_search is True:
        await tldr.apply_research()

    # Synthesize content
    await tldr.integrate_summaries()

    # Rewrite for response type and tone
    if tldr.polish is True:
        await tldr.polish_response()
        final_text = tldr.polished_summary
    else:
        final_text = tldr.executive_summary

    # Save final context
    if tldr.pdf is True:
        await tldr.save_to_pdf(final_text)

    # End session 
    await tldr.finish_session()


def cli_main():
    """Command-line entry point to run the tldr script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
