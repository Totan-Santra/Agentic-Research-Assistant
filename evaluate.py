import os
from dotenv import load_dotenv

from langsmith import Client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load .env
load_dotenv()

# -------------------------------------------------
# 1. Check API Keys
# -------------------------------------------------

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing in .env")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY is missing in .env")


# -------------------------------------------------
# 2. LangSmith Client
# -------------------------------------------------

client = Client()

DATASET_NAME = "Agentic-Research-Assistant-Evaluation"


# -------------------------------------------------
# 3. Evaluation Questions
# -------------------------------------------------

examples = [
    {
        "question": "What is the Transformer architecture?",
        "expected": "The Transformer is a neural network architecture based mainly on attention mechanisms."
    },
    {
        "question": "What is self-attention?",
        "expected": "Self-attention allows tokens to attend to other tokens in the same sequence."
    },
    {
        "question": "What is multi-head attention?",
        "expected": "Multi-head attention uses multiple attention heads to learn different relationships between tokens."
    },
    {
        "question": "What is positional encoding?",
        "expected": "Positional encoding provides information about the position of tokens in a sequence."
    },
    {
        "question": "What is RAG?",
        "expected": "RAG combines information retrieval with a language model to generate answers using retrieved context."
    },
    {
        "question": "What is 20 * 7?",
        "expected": "140"
    },
    {
        "question": "What is the difference between RAG and an LLM?",
        "expected": "An LLM generates text from learned parameters, while RAG retrieves external information and provides it to the LLM."
    },
    {
        "question": "Why is attention useful in Transformers?",
        "expected": "Attention helps the model identify relationships between different tokens in a sequence."
    },
    {
        "question": "What is an agent?",
        "expected": "An agent can reason about a task and decide which tools or actions to use."
    },
    {
        "question": "What is LangGraph?",
        "expected": "LangGraph is a framework for building stateful, multi-step agent workflows using graphs."
    }
]


# -------------------------------------------------
# 4. Create Dataset
# -------------------------------------------------

try:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for Agentic Research Assistant"
    )

    print("New LangSmith dataset created.")

    client.create_examples(
        inputs=[{"question": item["question"]} for item in examples],
        outputs=[{"expected": item["expected"]} for item in examples],
        dataset_id=dataset.id
    )

except Exception:
    print("Dataset already exists. Using existing dataset.")

    dataset = next(
        client.list_datasets(dataset_name=DATASET_NAME),
        None
    )

    if dataset is None:
        raise ValueError("Could not find or create LangSmith dataset.")


# -------------------------------------------------
# 5. Load Agent
# -------------------------------------------------

print("\nLoading Agent...")

from main import app

print("Agent loaded successfully.")


# -------------------------------------------------
# 6. Run Agent
# -------------------------------------------------

import uuid

def run_agent(inputs, **kwargs):

    question = inputs["question"]

    # Every evaluation question gets a fresh conversation
    thread_id = f"evaluation_{uuid.uuid4().hex}"

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return {
        "answer": result["messages"][-1].content
    }


# -------------------------------------------------
# 7. LLM Evaluator
# -------------------------------------------------

evaluator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def evaluate_answer(run, example):

    question = example.inputs["question"]
    expected = example.outputs["expected"]

    actual = run.outputs.get("answer", "")

    prompt = f"""
You are evaluating an AI agent.

Question:
{question}

Expected answer:
{expected}

Actual answer:
{actual}

Evaluate whether the actual answer correctly answers the question.

Give a score from 0 to 1.

0 = Completely incorrect
0.5 = Partially correct
1 = Correct

Return ONLY the number.
"""

    response = evaluator_llm.invoke(prompt)

    try:
        score = float(response.content.strip())
    except ValueError:
        score = 0.0

    return {
        "key": "correctness",
        "score": score
    }


# -------------------------------------------------
# 8. Run Evaluation
# -------------------------------------------------

print("\n" + "=" * 60)
print("STARTING LANGSMITH EVALUATION")
print("=" * 60)

results = client.evaluate(
    run_agent,
    data=DATASET_NAME,
    evaluators=[evaluate_answer],
    experiment_prefix="Agentic-Research-Assistant"
)


# -------------------------------------------------
# 9. Finished
# -------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nOpen LangSmith to see:")
print("- Questions")
print("- Agent answers")
print("- Correctness scores")
print("- Traces")
print("- Tool calls")
print("- Evaluation results")
