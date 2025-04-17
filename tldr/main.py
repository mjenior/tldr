#!/usr/bin/env python3

import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, function_tool

from .utils import read_yaml_config


async def main():
    msg = input("Hi! What is the project task or goal you'd like to address? ")

    # Run the entire orchestration in a single trace
    with trace("Project Orchestrator"):

        requester_result = await Runner.run(requester_agent, msg)
        print(f"\n\nRefined user request:\n{requester_result.final_output}")

        # Run the orchestrator agent with the refined user request
        orchestrator_result = await Runner.run(orchestrator_agent, requester_result.final_output)
        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Task step: {text}")

        # Assuming an agent that composes final project summaries might exist
        summarizer_result = await Runner.run(
            project_summarizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal project summary:\n{summarizer_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
