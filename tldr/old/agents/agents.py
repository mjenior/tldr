

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

from tldr.tools import *
from agents.tool import WebSearchTool

import pyaml


def create_agent(name_str, instructions_str, tools_lst, llm="gpt-4o")
    """Shortcut function to more easily generate agents on the fly"""
    try:
        global client
        assistant = client.beta.assistants.create(
            name=name_str,
            instructions=instructions_str,
            tools=tools_lst,
            model=llm)
        print(f"{name_str} agent created with ID: {assistant.id}")
    except Exception as e:
        print(f"Error creating {name_str} agent: {e}")

    return assistant




requester_instructions = """

"""



# Extrapolte details required in background research request
requester_agent = Agent(
    name="Project Summarizer Agent",
    instructions=requester_instructions,
    handoffs = [literature_collector])




collector_handoff = """

handoff_description: str | None = None
    A description of the agent. This is used when the agent is used as a handoff, so that an
    LLM knows what it does and when to invoke it.
    

"""


collector_guardrails = """

output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    A list of checks that run on the final output of the agent, after generating a response.
    Runs only if the agent produces a final output.

"""

# TODO: pull url text agent

# Should rank sources???
literature_collector = Agent(
    name = "Literature Assembly Agent",
    tools = [find_readable_files, 
    FileSearchTool(
                max_num_results=3,
                vector_store_ids=["vs1"],
                include_search_results=True,
            )],
    instructions = collector_instructions,
    handoffs = [research_gapfiller],
    handoff_description = collector_handoff,
    output_guardrails = collector_guardrails)




research_gapfiller = Agent(
    name = "Literature Assembly Agent",
    tools = [WebSearchTool],
    instructions = gapfiller_instructions,
    handoffs = [rag_specialist],
    handoff_description = gapfiller_handoff,
    output_guardrails = gapfiller_guardrails)



# TODO:
# RAG specialist
# Science validation agent: LLMs as tools with topic experts

# Needs expanding
report_generator = Agent(
    name="Summarizer Agent",
    instructions=(
        "You are responsible for synthesizing and summarizing the outputs of project tasks "
        "into a coherent summary. Ensure all important aspects of the project are covered."
    )
)




# --- ORCHESTRATION ------------------------------------------------------------------------------------ #


# TODO: 
# - Biology orchestrator
# - Analysis orchestrator
# - Writing orchestrator

software_orchestrator = Agent(
    name="Orchestrator Agent",
    instructions=software_orchestrator_instructions,

    # TODO: handoffs

    tools=[
        product_manager.as_tool(
            tool_name="manage_project",
            tool_description="Coordinate high-level project goals and requirements."
        ),
        architect_agent.as_tool(
            tool_name="design_system_architecture",
            tool_description="Design the system architecture for the project."
        ),
        task_master.as_tool(
            tool_name="generate_tasks",
            tool_description="Generate tasks from user requirements."
        ),
        backend_developer.as_tool(
            tool_name="develop_backend",
            tool_description="Implement server-side functionality and design system architecture."
        ),
        frontend_developer.as_tool(
            tool_name="develop_frontend",
            tool_description="Implement and test UI and client-facing components."
        ),
    ],

    # TODO: input_guardrails

    # TODO: output_guardrails

)



# --- MANAGEMENT ------------------------------------------------------------------------------------ #





