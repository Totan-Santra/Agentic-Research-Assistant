import os
import ast
import operator
import sqlite3
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


# ============================================================
# LANGSMITH CONFIGURATION
# ============================================================

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Agentic-Research-Assistant"

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY


# ============================================================
# API KEY CHECK
# ============================================================

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env file")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing in .env file")

if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY is missing in .env file")


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_completion_tokens=256,
    reasoning_effort="low",
)


# ============================================================
# RAG DATABASE
# ============================================================

print("\nLoading RAG database...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="Data",
    persist_directory="./Chromadb",
    embedding_function=embedding_model,
)

print("RAG database loaded successfully.")


# ============================================================
# RAG TOOL
# ============================================================

@tool
def retrieve(query: str) -> str:
    """
    Retrieve relevant information from the local knowledge base.
    Use this tool for questions related to the stored documents.
    """

    # No extra LLM query rewriting.
    # This saves tokens and reduces unnecessary API calls.

    rewritten_query = query

    print(f"\n[RAG QUERY] {rewritten_query}")

    # Retrieve only 3 documents
    results = vectorstore.similarity_search_with_score(
        rewritten_query,
        k=3
    )

    output = []

    for doc, score in results:

        # Keep only reasonably relevant documents
        if score < 1.0:

            # Limit each document size
            content = doc.page_content[:1500]

            output.append(content)

    if not output:
        return "No relevant information found in the knowledge base."

    # Compact RAG context
    return "\n\n---\n\n".join(output)


# ============================================================
# SAFE CALCULATOR
# ============================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(node):

    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Invalid number")

    if isinstance(node, ast.BinOp):

        left = safe_calculate(node.left)
        right = safe_calculate(node.right)

        operator_type = type(node.op)

        if operator_type not in allowed_operators:
            raise ValueError("Operator not allowed")

        return allowed_operators[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):

        operand = safe_calculate(node.operand)

        operator_type = type(node.op)

        if operator_type not in allowed_operators:
            raise ValueError("Operator not allowed")

        return allowed_operators[operator_type](operand)

    raise ValueError("Invalid expression")


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    """

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = safe_calculate(tree.body)

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"


# ============================================================
# TAVILY WEB SEARCH
# ============================================================

tavily_search = TavilySearch(
    max_results=3
)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current, latest, or real-time information.
    """

    print(f"\n[WEB SEARCH] {query}")

    try:

        result = tavily_search.invoke(
            {"query": query}
        )

        # Convert result to text
        result_text = str(result)

        # Limit web-search context
        # This prevents very large requests to the LLM.
        result_text = result_text[:4000]

        return result_text

    except Exception as e:

        return f"Web search error: {str(e)}"


# ============================================================
# TOOLS
# ============================================================

tools = [
    retrieve,
    calculator,
    web_search
]


print("\nAvailable tools:")

for t in tools:
    print(f"- {t.name}")


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools)

print("\n========== TOOL CHECK ==========")

for t in tools:
    print("Tool object:", t)
    print("Tool name:", getattr(t, "name", None))
    print("Tool description:", getattr(t, "description", None))
    print("-------------------------------")


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent Agentic Research Assistant.

You have access to three tools:

1. retrieve
   Use this tool when the question is related to information
   contained in the local knowledge base.

2. calculator
   Use this tool for mathematical calculations.

3. web_search
   Use this tool when the user asks for current, latest,
   recent, real-time, or web-based information.

Tool selection rules:

- Do not use RAG for simple mathematics.
- Use calculator for mathematical calculations.
- Use web_search for latest/current information.
- Use retrieve for questions related to the stored documents.
- If a tool is not necessary, answer directly.

Always provide a clear and concise final answer.
"""


# ============================================================
# CHATBOT NODE
# ============================================================

def chatbot(state: State):

    messages = state["messages"]

    # ========================================================
    # MEMORY OPTIMIZATION
    # ========================================================
    # Only send the most recent 6 messages to the LLM.
    #
    # The complete conversation is still stored in SQLite,
    # but we don't send the entire history to Groq every time.
    # This reduces token usage and improves response speed.
    # ========================================================

    recent_messages = messages[-6:]

    messages_with_system = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] + recent_messages

    # ========================================================
    # LLM CALL
    # ========================================================

    response = llm_with_tools.invoke(
        messages_with_system
    )

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# ROUTER
# ============================================================

def router(state: State):

    messages = state["messages"]

    last_message = messages[-1]

    # If LLM wants to call a tool
    if hasattr(last_message, "tool_calls"):

        if last_message.tool_calls:
            return "tools"

    return END


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


# Add nodes
graph.add_node(
    "chatbot",
    chatbot
)

graph.add_node(
    "tools",
    tool_node
)


# Start → chatbot
graph.add_edge(
    START,
    "chatbot"
)


# chatbot → tools OR END
graph.add_conditional_edges(
    "chatbot",
    router,
    {
        "tools": "tools",
        END: END
    }
)


# tools → chatbot
graph.add_edge(
    "tools",
    "chatbot"
)


# ============================================================
# PERSISTENT MEMORY
# ============================================================

MEMORY_DATABASE = "conversation_memory.sqlite"

conn = sqlite3.connect(
    MEMORY_DATABASE,
    check_same_thread=False
)

memory = SqliteSaver(conn)


# ============================================================
# COMPILE APPLICATION
# ============================================================

app = graph.compile(
    checkpointer=memory
)

print("\nLangGraph application compiled successfully.")


# ============================================================
# MAIN CHAT FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("🤖 AGENTIC RESEARCH ASSISTANT")
    print("=" * 60)

    print("\nType 'exit' to stop.")

    # Persistent conversation thread
    THREAD_ID = "totan_session_1"

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        }
    }

    while True:

        try:

            user_input = input("\nYou: ")

            # Exit
            if user_input.lower() in [
                "exit",
                "quit",
                "bye"
            ]:

                print("\nGoodbye! 👋")
                break

            # Empty input
            if not user_input.strip():
                continue

            # =================================================
            # RUN LANGGRAPH
            # =================================================

            result = app.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=config
            )

            # =================================================
            # FINAL RESPONSE
            # =================================================

            final_message = result["messages"][-1]

            print("\nAssistant:")

            print(
                final_message.content
            )

        except KeyboardInterrupt:

            print("\n\nProgram stopped.")
            break

        except Exception as e:

            print(
                f"\nError: {str(e)}"
            )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()

