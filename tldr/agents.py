
from biosquad.tools import *
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

from agents.tool import FileSearchTool, WebSearchTool

import pyaml






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


# Should rank sources???
literature_collector = Agent(
    name = "Literature Assembly Agent",
    tools = [find_readable_files, FileSearchTool],
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
# Gapfiller 
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





