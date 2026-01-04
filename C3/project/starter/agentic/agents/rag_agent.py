import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState


class RAGState(MessagesState):
    """
    State definition for the RAG Agent.
    """
    question: str
    documents: List[Document]
    answer: str


class RAGAgent:
    def __init__(self, vector_store: Chroma, model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG Agent with a vector store and LLM.

        Args:
            vector_store (Chroma): The initialized vector store containing the documents.
            model_name (str): The name of the OpenAI model to use.
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.graph = self._build_graph()

    def retrieve(self, state: RAGState):
        """
        Retrieve relevant documents from the vector store based on the question.
        """
        question = state["question"]
        retrieved_docs = self.vector_store.similarity_search(question)
        return {"documents": retrieved_docs}

    def augment(self, state: RAGState):
        """
        Augment the prompt with the retrieved context.
        """
        question = state["question"]
        documents = state["documents"]
        docs_content = "\n\n".join(doc.page_content for doc in documents)

        template = ChatPromptTemplate([
            ("system", "You are an assistant for question-answering tasks."),
            ("human", "Use the following pieces of retrieved context to answer the question. "
                      "If you don't know the answer, just say that you don't know. "
                      "Use three sentences maximum and keep the answer concise. "
                      "\n# Question: \n-> {question} "
                      "\n# Context: \n-> {context} "
                      "\n# Answer: "),
        ])

        messages = template.invoke(
            {"context": docs_content, "question": question}
        ).to_messages()

        return {"messages": messages}

    def generate(self, state: RAGState):
        """
        Generate the final answer using the LLM.
        """
        ai_message = self.llm.invoke(state["messages"])
        return {"answer": ai_message.content, "messages": ai_message}

    def _build_graph(self):
        """
        Compile the StateGraph for the RAG workflow.
        """
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("augment", self.augment)
        workflow.add_node("generate", self.generate)

        # Add edges
        # Flow: Start -> Retrieve -> Augment -> Generate -> End
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "augment")
        workflow.add_edge("augment", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def get_graph(self):
        """
        Return the compiled graph runnable.
        """
        return self.graph