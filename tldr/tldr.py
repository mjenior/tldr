#!/usr/bin/env python3

import asyncio

from .core import TldrEngine
from .utils import parse_user_arguments


async def main():
    """Main entry point for the tldr command"""

    # Read in content and extend user query
    engine_args = parse_user_arguments()
    tldr = TldrEngine(**engine_args)

    # Report something if verbose is False
    if tldr.verbose is False:
        print("TLDR is working, this may take a minute...")

    # Extend user query
    if len(tldr.query) > 0 and tldr.refine_query is True:
        tldr.logger.info("Refining user query...")
        await tldr.refine_user_query()
        tldr.logger.info("Querying local embedded text vectors...")
        embedded_context = await tldr.get_embedded_context()
        tldr.embedded_context = "\n".join(embedded_context.values())
        # TODO: Add a score cutoff or something, maybe just report them...

    # Condense any context docs provided
    if tldr.raw_context is not None:
        tldr.logger.info("Refining background context...")
        await tldr.format_context_references()

    # Summarize documents
    tldr.logger.info("Generating summaries for selected resources...")
    await tldr.summarize_resources()

    # Synthesize content
    if len(tldr.reference_summaries) >= 2:
        await tldr.integrate_summaries()
    else:
        tldr.executive_summary = tldr.reference_summaries[0]

    # Use research agent to fill gaps
    if tldr.web_search:
        await tldr.apply_research()

    # Rewrite for response type and tone
    await tldr.polish_response()
    tldr.logger.info(tldr.polished_summary)

    # Complete run - Move logile and report usage
    tldr.format_spending()
    tldr.generate_token_report()


if __name__ == "__main__":
    asyncio.run(main())
