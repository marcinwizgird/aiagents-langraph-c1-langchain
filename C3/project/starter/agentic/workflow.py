import asyncio
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import from the Adapter (or your local mcp_client if you prefer)
from langchain_mcp_adapters.client import MultiServerMCPClient

from agentic.mcp_config import MCP_SERVERS
from agentic.agents import SupervisorAgent, ToolAgent, AgentState

# Global reference to keep the client connection alive
_client_instance = None


async def initialize_workflow():
    """
    Async Factory: Connects to MCP, fetches tools, and builds the graph.
    Returns: (orchestrator_graph, client_instance)
    """
    global _client_instance

    print("🔌 Connecting to MCP Server...")
    # 1. Initialize and Connect Client
    # We use the config from mcp_config.py
    _client_instance = MultiServerMCPClient(MCP_SERVERS)
    await _client_instance.__aenter__()

    # 2. Fetch All Tools
    # Assuming 'unified_service' is the key in your MCP_SERVERS config
    # If using the '3 server' config, call get_tools for each key.
    # Here we assume the single unified server approach for simplicity.
    all_tools = await _client_instance.get_tools("unified_service")

    # 3. Filter Tools for Specialization
    # We manually assign tools to agents based on function names
    sub_tool_names = ["lookup_customer", "get_user_subscription", "cancel_subscription_action"]
    res_tool_names = ["lookup_customer", "get_available_experiences", "get_user_reservations",
                      "create_reservation_action"]
    kb_tool_names = ["search_knowledge_base"]

    sub_tools = [t for t in all_tools if t.name in sub_tool_names]
    res_tools = [t for t in all_tools if t.name in res_tool_names]
    kb_tools = [t for t in all_tools if t.name in kb_tool_names]

    print(f"✅ Tools Assigned: Subscription={len(sub_tools)}, Reservation={len(res_tools)}, Knowledge={len(kb_tools)}")

    # 4. Instantiate Agents
    supervisor = SupervisorAgent()

    sub_agent = ToolAgent(
        name="Subscription",
        tools=sub_tools,
        system_prompt="You are the Subscription Manager. Always verify user identity by email first using `lookup_customer`. Do not guess IDs."
    ).get_graph()

    res_agent = ToolAgent(
        name="Reservation",
        tools=res_tools,
        system_prompt="You are the Reservation Specialist. If a user asks to book, list `get_available_experiences` first."
    ).get_graph()

    kb_agent = ToolAgent(
        name="Knowledge",
        tools=kb_tools,
        system_prompt="You are a Support Assistant. Use `search_knowledge_base` to answer questions based on official policy."
    ).get_graph()

    # 5. Build the Master Graph
    workflow = StateGraph(AgentState)

    # Define Nodes
    def triage_node(state):
        return {"next_step": supervisor.route(state["messages"])}

    def sub_node(state):
        result = sub_agent.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}

    def res_node(state):
        result = res_agent.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}

    def kb_node(state):
        result = kb_agent.invoke(state)
        return {"messages": result["messages"][len(state["messages"]):]}

    def escalate_node(state):
        return {"messages": [
            HumanMessage(content="I have escalated your ticket to a human agent. They will contact you shortly.",
                         name="System")]}

    # Add Nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("manage_subscription", sub_node)
    workflow.add_node("manage_reservations", res_node)
    workflow.add_node("consult_kb", kb_node)
    workflow.add_node("escalate", escalate_node)

    # Add Edges
    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", lambda x: x["next_step"], {
        "manage_subscription": "manage_subscription",
        "manage_reservations": "manage_reservations",
        "consult_kb": "consult_kb",
        "escalate": "escalate"
    })

    # Return to END (User Input)
    for node in ["manage_subscription", "manage_reservations", "consult_kb", "escalate"]:
        workflow.add_edge(node, END)

    # Compile
    orchestrator = workflow.compile(checkpointer=MemorySaver())

    return orchestrator