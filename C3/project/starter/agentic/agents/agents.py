from typing import List, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- Output Models ---
class RouteDecision(BaseModel):
    next_step: Literal["manage_subscription", "manage_reservations", "consult_kb", "escalate"] = Field(
        description="The department to route the ticket to."
    )


# --- Shared State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: str


# --- 1. Generic Tool Agent (For Sub/Res/KB) ---
class ToolAgent:
    def __init__(self, name: str, tools: list, system_prompt: str, model_name: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(self.tools)
        self.graph = self._build_graph()

    def reason(self, state: AgentState):
        msgs = [SystemMessage(content=self.system_prompt)] + state["messages"]
        return {"messages": [self.llm.invoke(msgs)]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("reason", self.reason)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "reason")
        workflow.add_conditional_edges("reason", lambda s: "tools" if s["messages"][-1].tool_calls else END,
                                       {"tools": "tools", END: END})
        workflow.add_edge("tools", "reason")
        return workflow.compile()

    def get_graph(self): return self.graph


# --- 2. Supervisor Agent ---
class SupervisorAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.system_prompt = """Route the user's request.
        1. manage_subscription: Billing, plans, cancellation, account status.
        2. manage_reservations: Booking events, checking availability.
        3. consult_kb: General help, policies, login issues.
        4. escalate: Complex technical issues or angry users."""

    def route(self, messages: list) -> str:
        structured_llm = self.llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke([SystemMessage(content=self.system_prompt), messages[-1]])
        return decision.next_step