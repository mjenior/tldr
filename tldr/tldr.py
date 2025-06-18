#!/usr/bin/env python3

import asyncio

from .core import TldrEngine
from .args import parse_tldr_arguments


async def main():
    """Main entry point for the tldr command"""

    # Initialize TldrEngine and load content
    tldr = TldrEngine(**parse_tldr_arguments())
    await tldr.initialize_async_session()

    # Extend user query if needed
    if tldr.refine_query:
        await tldr.refine_user_query()

    # Summarize documents
    await tldr.summarize_resources()

    # Use research agent to fill gaps
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
