## Implementation Decisions
## Remark: The following documentations explains the implementation decisions which have been made while creating project decisions.
## Remark: The follwing project leaves little space for architecture innovation.


### 1. Multi-Agent Architecture with LangGraph
The core of the application is a state machine defined using `LangGraph`.
* **Routing:** A `classify_intent` node analyzes the user's input and conversation history to determine the intent (`qa`, `summarization`, `calculation`, or `unknown`).
* **Specialized Agents:** Based on the intent, the workflow routes execution to one of three specialized nodes: `qa_agent`, `summarization_agent`, or `calculation_agent`.
* **Unified Memory Update:** All agent paths converge on an `update_memory` node, ensuring the conversation state is consistently maintained regardless of the path taken.
### the Agent concept is a LangGraph node that encapsulates the logic for a specific task with function what constitutes a specific form of agent.

### 2. Structured Outputs
Instead of parsing raw string responses, the system leverages `llm.with_structured_output()` to force the LLM to return data adhering to defined Pydantic models.
* **Benefits:** This guarantees that downstream functions receive valid JSON-like objects with expected fields (e.g., `confidence` scores, `source` lists).
* **Schemas:** Defined in `schemas.py`, including `AnswerResponse`, `SummarizationResponse`, `CalculationResponse`, and `UserIntent`.

### 3. State and Memory Management
The system uses a persistent `AgentState` object that flows through the graph.
* **State Schema:** The `AgentState` tracks the `user_input`, `messages` history, `active_documents`, and a cumulative log of `actions_taken` (using a reducer).
* **Persistence:** A `MemorySaver` checkpointer is compiled into the workflow. This allows the system to "remember" previous turns by reloading the state associated with a specific `session_id` (thread).
* **Session Storage:** Session metadata and history are also serialized to JSON files in the `./sessions` directory for long-term storage and recovery.

### 4. Tool Safety
The `calculator` tool in `tools.py` implements a restricted `eval()` environment.
* **Validation:** Input strings are checked against an allowed set of characters (digits and basic operators).
* **Restricted Scope:** The `eval()` function runs with `__builtins__` set to `None`, granting access only to specific safe math functions like `abs`, `round`, and `min`.

## Setup and Usage

### Prerequisites
* Python 3.10+
* OpenAI API Key

### Installation
1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_api_key_here
    ```

### Running the Assistant
Execute the main script to start the interactive CLI:
```bash
python main.py
Example Conversations
Scenario 1: Question Answering (Q&A)
User: "Who is the client for invoice INV-001?" Assistant: "The client for invoice INV-001 is Acme Corporation."

Mechanism: classify_intent -> qa_agent (uses document_search tool) -> update_memory

Scenario 2: Calculation
User: "Calculate the total value if we add the amount from invoice INV-001 to the contract CON-001." Assistant: "The total value is $200,000. This is calculated by adding the invoice total ($20,000) to the contract value ($180,000)."

Mechanism: classify_intent -> calculation_agent (uses document_reader to get values, then calculator tool) -> update_memory

Scenario 3: Summarization
User: "Summarize the service agreement CON-001." Assistant: "The Service Agreement (CON-001) is between DocDacity Solutions and Healthcare Partners LLC. It covers document processing, support, and analytics for a 12-month term with a total value of $180,000."

Mechanism: classify_intent -> summarization_agent (uses document_reader) -> update_memory

Project Structure
src/agent.py: Defines the LangGraph workflow and agent nodes.

src/assistant.py: Manages the application session and main processing loop.

src/prompts.py: Contains system prompts and template logic.

src/schemas.py: Defines Pydantic models for structured data.

src/tools.py: Implements the calculator and document tools.

src/retrieval.py: Simulates the document database.