# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
)
import gc

from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path, echo=False):
    # ... existing code ...
    # Remove the file if it exists
    if os.path.exists(db_path):
        # Force garbage collection to close any dangling connections
        gc.collect()
        try:
            os.remove(db_path)
            print(f"✅ Removed existing {db_path}")
        except PermissionError:
            print(f"⚠️ Could not remove {db_path}. It might be in use.")
            print("Attempting to wait and retry...")
            time.sleep(1)
            try:
                os.remove(db_path)
                print(f"✅ Removed existing {db_path} after retry")
            except PermissionError:
                 print(f"❌ Failed to remove {db_path}. Please restart the kernel to release the file lock.")
                 raise


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(agent:CompiledStateGraph, ticket_id:str):
    is_first_iteration = False
    messages = [SystemMessage(content = f"ThreadId: {ticket_id}")]
    while True:
        user_input = input("User: ")
        print("User:", user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        messages = [HumanMessage(content=user_input)]
        if is_first_iteration:
            messages.append(HumanMessage(content=user_input))
        trigger = {
            "messages": messages
        }
        config = {
            "configurable": {
                "thread_id": ticket_id,
            }
        }
        
        result = agent.invoke(input=trigger, config=config)
        print("Assistant:", result["messages"][-1].content)
        is_first_iteration = False