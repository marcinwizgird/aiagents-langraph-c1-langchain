from typing import Literal, TypedDict, Annotated, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- Output Models ---
class RouteDecision(BaseModel):
    next_step: Literal["manage_subscription", "manage_reservations", "consult_kb", "escalate"] = Field(
        description="The specific department to route the ticket to."
    )


# --- Shared State ---
class AgentState(TypedDict):
    # Standard LangGraph message history
    messages: Annotated[List[BaseMessage], add_messages]
    # Contextual data
    user_id: Optional[str]
    user_info: Optional[dict]  # Long-term memory context
    ticket_status: Optional[str]
    next_step: Optional[str]


# --- 1. Generic Tool Agent ---
# This class creates a specialized agent (Sub, Res, KB) based on the tools injected
class ToolAgent:
    def __init__(self, name: str, tools: list, system_prompt: str, model_name: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(self.tools)
        self.graph = self._build_graph()

    def reason(self, state: AgentState):
        """Determines whether to call a tool or answer the user."""
        msgs = [SystemMessage(content=self.system_prompt)] + state["messages"]
        return {"messages": [self.llm.invoke(msgs)]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("reason", self.reason)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "reason")

        # Conditional Edge: If tool calls exist, go to 'tools', else END
        def should_continue(state):
            last_msg = state["messages"][-1]
            return "tools" if last_msg.tool_calls else END

        workflow.add_conditional_edges("reason", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "reason")  # Loop back to reasoning after tool execution
        return workflow.compile()

    def get_graph(self): return self.graph


# --- 2. Supervisor Agent ---
class RouteDecision(BaseModel):
    next_step: Literal[
        "manage_bookings",
        "manage_reservations",
        "manage_account",
        "consult_kb",
        "escalate"
    ] = Field(description="The specific department to route the ticket to.")


class SupervisorAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.system_prompt = """You are the Triage Supervisor for CultPass.
        Analyze the conversation and route the ticket to the correct specialist.

        1. manage_bookings: For creating NEW bookings, browsing available classes/experiences, or checking schedules.
        2. manage_reservations: For modifying, cancelling, or viewing EXISTING reservations/bookings.
        3. manage_account: For subscription status, billing, password resets, or profile updates.
        4. consult_kb: For general questions, policies, opening hours, or "how-to" guides.
        5. escalate: If the user is angry, using profanity, or the request is clearly outside the scope of the automated system.
        """

    def route(self, messages: list) -> str:
        structured_llm = self.llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke([SystemMessage(content=self.system_prompt)] + messages)
        return decision.next_step


# --- 3. Supervisor Agent ---

class RAGState(TypedDict):
    """Internal state for the RAG loop."""
    query: str
    documents: List[str]
    generation: str
    evaluation: str
    attempt: int
    final_answer: str


class RAGAgent:
    def __init__(self, tools: list, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        # We assume tools[0] is the retrieval tool (e.g., search_kb)
        self.retriever_tool = tools[0]
        self.max_retries = 3
        self.graph = self._build_graph()

    def _reformulate_query(self, state: RAGState):
        """Rewrites the query to be more search-friendly based on previous failure."""
        prompt = f"Reword this query to be more specific for a vector search engine: {state['query']}"
        response = self.llm.invoke(prompt)
        return {"query": response.content, "attempt": state["attempt"] + 1}

    def _retrieve(self, state: RAGState):
        """Calls the retrieval tool."""
        # This assumes the tool returns a string or list of strings
        docs = self.retriever_tool.invoke(state['query'])
        # Normalize to list
        if isinstance(docs, str):
            docs = [docs]
        return {"documents": docs}

    def _generate(self, state: RAGState):
        """Generates an answer based on retrieved docs."""
        context = "\n".join(state['documents'])
        prompt = f"Context: {context}\n\nQuestion: {state['query']}\n\nAnswer the question using ONLY the context."
        response = self.llm.invoke(prompt)
        return {"generation": response.content}

    def _evaluate(self, state: RAGState):
        """Evaluates if the answer is satisfactory."""
        # Simple self-reflection check
        prompt = f"Question: {state['query']}\nAnswer: {state['generation']}\n\nIs this answer helpful and relevant? Respond with 'YES' or 'NO'."
        response = self.llm.invoke(prompt)
        return {"evaluation": response.content.strip().upper()}

    def _check_quality(self, state: RAGState):
        if "YES" in state["evaluation"]:
            return "good"
        if state["attempt"] >= self.max_retries:
            return "max_retries"
        return "bad"

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("evaluate", self._evaluate)
        workflow.add_node("reformulate", self._reformulate_query)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._check_quality,
            {
                "good": END,
                "max_retries": END,
                "bad": "reformulate"
            }
        )

        workflow.add_edge("reformulate", "retrieve")

        return workflow.compile()

    def as_node(self):
        """Wraps the internal RAG graph to function as a node in the main AgentState graph."""

        def _call(state: AgentState):
            # 1. Map AgentState to RAGState
            last_message = state["messages"][-1].content
            initial_input = {"query": last_message, "attempt": 0, "documents": [], "generation": "", "evaluation": "",
                             "final_answer": ""}

            # 2. Run internal loop
            result = self.graph.invoke(initial_input)

            # 3. Map result back to AgentState (as an AI Message)
            final_content = result.get("generation", "I couldn't find a good answer in the knowledge base.")
            return {"messages": [AIMessage(content=final_content, name="KnowledgeAgent")]}

        return _call