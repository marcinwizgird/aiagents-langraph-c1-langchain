import asyncio
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# UPDATE: Import from the adapter package instead of local file
from langchain_mcp_adapters.client import MultiServerMCPClient

from agentic.mcp_config import MCP_SERVERS
from agentic.agents import SupervisorAgent, ToolAgent


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: str


async def initialize_workflow():
    """Initializes clients, loads tools, and builds the graph."""

    # 1. Initialize MCP Client using the Adapter
    # We pass the configuration dict directly
    client = MultiServerMCPClient(MCP_SERVERS)
    await client.__aenter__()

    # 2. Get Tools per Agent
    # Note: Assuming the adapter API supports fetching by server name key
    # If the adapter returns all tools flatly, we might need to inspect tool names
    sub_tools = await client.get_tools(server_name="subscription")
    res_tools = await client.get_tools(server_name="reservation")
    kb_tools = await client.get_tools(server_name="knowledge")

    print(f"✅ Workflow Init: SubTools={len(sub_tools)}, ResTools={len(res_tools)}, KBTools={len(kb_tools)}")

    # 3. Instantiate Agents
    supervisor = SupervisorAgent()

    sub_agent = ToolAgent(
        name="Subscription",
        tools=sub_tools,
        system_prompt="You are the Subscription Manager. Always verify user identity by email first. Tools: lookup_customer, get_subscription_details, cancel_subscription_action."
    ).get_graph()

    res_agent = ToolAgent(
        name="Reservation",
        tools=res_tools,
        system_prompt="You are the Reservation Specialist. List available experiences before booking. Tools: get_available_experiences, create_reservation_action."
    ).get_graph()

    kb_agent = ToolAgent(
        name="Knowledge",
        tools=kb_tools,
        system_prompt="You are a Support Assistant. Use `search_knowledge_base` to answer questions based on policy."
    ).get_graph()

    # 4. Build Graph
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("triage", lambda s: {"next_step": supervisor.route(s["messages"])})
    workflow.add_node("manage_subscription",
                      lambda s: {"messages": sub_agent.invoke(s)["messages"][len(s["messages"]):]})
    workflow.add_node("manage_reservations",
                      lambda s: {"messages": res_agent.invoke(s)["messages"][len(s["messages"]):]})
    workflow.add_node("consult_kb", lambda s: {"messages": kb_agent.invoke(s)["messages"][len(s["messages"]):]})
    workflow.add_node("escalate",
                      lambda s: {"messages": [HumanMessage(content="Escalating to human agent.", name="System")]})

    # Edges
    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", lambda x: x["next_step"], {
        "manage_subscription": "manage_subscription",
        "manage_reservations": "manage_reservations",
        "consult_kb": "consult_kb",
        "escalate": "escalate"
    })

    for node in ["manage_subscription", "manage_reservations", "consult_kb", "escalate"]:
        workflow.add_edge(node, END)

    return workflow.compile(checkpointer=MemorySaver())