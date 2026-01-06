import sys
import os
import uuid
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Add parent dir to path to find data.models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.models import User, UserMemory, Subscription, Experience, Reservation

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../../data/external/cultpass.db")
VECTOR_PATH = os.path.join(BASE_DIR, "../../data/vector_store")

engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)

mcp = FastMCP("CultPass Service")

# --- Context / Memory Tools ---
@mcp.tool()
def get_user_context(user_id: str) -> str:
    """Get long-term memory/preferences for a user."""
    session = Session()
    try:
        memories = session.query(UserMemory).filter(UserMemory.user_id == user_id).all()
        return "\n".join([f"- {m.content}" for m in memories]) if memories else ""
    finally:
        session.close()

# --- Subscription Tools ---
@mcp.tool()
def get_subscription(user_id: str) -> str:
    """Get subscription status."""
    session = Session()
    try:
        sub = session.query(Subscription).filter(Subscription.user_id == user_id).first()
        return f"Status: {sub.status} | Tier: {sub.tier}" if sub else "No subscription"
    finally:
        session.close()

# --- Reservation Tools ---
@mcp.tool()
def list_experiences() -> str:
    """List available experiences."""
    session = Session()
    try:
        exps = session.query(Experience).all()
        return "\n".join([f"ID: {e.experience_id} | {e.title} ({e.slots_available} slots)" for e in exps])
    finally:
        session.close()

# --- Knowledge Tools ---
@mcp.tool()
def search_kb(query: str) -> str:
    """Search knowledge base."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(collection_name="cultpass_knowledge", embedding_function=embeddings, persist_directory=VECTOR_PATH)
    docs = db.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No info found."

if __name__ == "__main__":
    mcp.run()