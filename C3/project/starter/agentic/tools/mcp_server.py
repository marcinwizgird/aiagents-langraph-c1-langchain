import sys
import os
import shutil
import uuid
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import models
from data.models.cultpass import User as CultUser, Subscription, Reservation, Experience
from data.models.udahub import Knowledge

# --- Database Setup ---
CULTPASS_DB_URL = "sqlite:///data/external/cultpass.db"
UDAHUB_DB_URL = "sqlite:///data/core/udahub.db"
VECTOR_DB_PATH = "data/vector_store"

cultpass_engine = create_engine(CULTPASS_DB_URL)
udahub_engine = create_engine(UDAHUB_DB_URL)
CultSession = sessionmaker(bind=cultpass_engine)
UdaSession = sessionmaker(bind=udahub_engine)

# Initialize FastMCP (Unified Service)
mcp = FastMCP("CultPass Unified Service")

# --- Helpers ---
def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="cultpass_knowledge",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

# --- Admin Utility (Optional: Run via `python mcp_server.py init_vector`) ---
if len(sys.argv) > 1 and sys.argv[1] == "init_vector":
    print("Initializing Vector DB...")
    session = UdaSession()
    try:
        articles = session.query(Knowledge).all()
        if articles:
            docs = [
                Document(page_content=f"Title: {a.title}\nContent: {a.content}\nTags: {a.tags}", metadata={"title": a.title, "id": a.article_id})
                for a in articles
            ]
            if os.path.exists(VECTOR_DB_PATH): shutil.rmtree(VECTOR_DB_PATH)
            get_vector_store().add_documents(docs)
            print(f"✅ Indexed {len(docs)} articles.")
    finally:
        session.close()
    sys.exit(0)

# ==========================================
# ALL TOOLS (No Conditional Compilation)
# ==========================================

# --- Shared / Customer Tools ---

@mcp.tool()
def lookup_customer(email: str) -> str:
    """Retrieves customer profile by email (ID, Name, Block Status)."""
    session = CultSession()
    try:
        user = session.query(CultUser).filter(CultUser.email == email).first()
        return f"User ID: {user.user_id}, Name: {user.full_name}, Blocked: {user.is_blocked}" if user else "User not found."
    finally:
        session.close()

# --- Subscription Tools ---

@mcp.tool()
def get_user_subscription(user_id: str) -> str:
    """Fetches subscription details (Tier, Status, Quota) for a user ID."""
    session = CultSession()
    try:
        sub = session.query(Subscription).filter(Subscription.user_id == user_id).first()
        return (f"Sub ID: {sub.subscription_id}, Status: {sub.status}, Tier: {sub.tier}, Quota: {sub.monthly_quota}"
                if sub else "No subscription found.")
    finally:
        session.close()

@mcp.tool()
def cancel_subscription_action(user_id: str) -> str:
    """Cancels a user's subscription immediately."""
    session = CultSession()
    try:
        sub = session.query(Subscription).filter(Subscription.user_id == user_id).first()
        if not sub: return "Subscription not found."
        sub.status = "cancelled"
        session.commit()
        return f"Success: Subscription {sub.subscription_id} cancelled."
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()

# --- Reservation Tools ---

@mcp.tool()
def get_available_experiences() -> str:
    """Lists upcoming experiences that have available slots."""
    session = CultSession()
    try:
        exps = session.query(Experience).filter(Experience.slots_available > 0).limit(5).all()
        if not exps: return "No experiences found."
        return "\n".join([f"- ID: {e.experience_id} | {e.title} | Slots: {e.slots_available}" for e in exps])
    finally:
        session.close()

@mcp.tool()
def get_user_reservations(user_id: str) -> str:
    """Lists existing reservations for a user."""
    session = CultSession()
    try:
        res_list = session.query(Reservation).filter(Reservation.user_id == user_id).all()
        if not res_list: return "No reservations found."
        return "\n".join([f"- ResID: {r.reservation_id} | ExpID: {r.experience_id} | Status: {r.status}" for r in res_list])
    finally:
        session.close()

@mcp.tool()
def create_reservation_action(user_id: str, experience_id: str) -> str:
    """Books an experience for a user if slots are available."""
    session = CultSession()
    try:
        exp = session.query(Experience).filter(Experience.experience_id == experience_id).first()
        if not exp or exp.slots_available < 1: return "Experience not found or fully booked."
        new_res = Reservation(reservation_id=str(uuid.uuid4())[:8], user_id=user_id, experience_id=experience_id, status="confirmed")
        exp.slots_available -= 1
        session.add(new_res)
        session.commit()
        return f"Success: Reservation {new_res.reservation_id} confirmed."
    except Exception as e:
        session.rollback()
        return f"Error: {str(e)}"
    finally:
        session.close()

# --- Knowledge Tools ---

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Semantic search for knowledge base articles."""
    try:
        db = get_vector_store()
        docs = db.similarity_search(query, k=3)
        if not docs: return "No relevant articles found."
        return "\n\n".join([f"--- {d.metadata.get('title')} ---\n{d.page_content}" for d in docs])
    except Exception as e:
        return f"Search Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")