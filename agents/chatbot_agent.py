# agents/chatbot_agent.py
# This agent handles conversation with the user.
# It answers questions about the dataset, explains concepts,
# and responds to anything that isn't cleaning/preprocessing/analysis.

import sys
import os
import logging
sys.path.append(os.getcwd())

import pandas as pd
from utils.llm_config import get_llm_client, MODEL_NAME

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
MAX_TOKENS       = 1024   # Hard cap on response length — prevents mid-word cutoff
MAX_HISTORY_TURNS = 10    # Keep only the last N conversation turns
                           # Prevents the context window from overflowing on long chats
MAX_DATASET_ROWS  = 3     # How many sample rows to show the AI


def _build_dataset_context(df: pd.DataFrame) -> str:
    """
    Builds a safe, size-limited summary of the dataset for the system prompt.
    Without limits, large files can overflow the context window.
    """
    if df is None or df.empty:
        return "No dataset has been uploaded yet."

    try:
        # Limit columns shown to avoid enormous prompts on wide datasets
        cols_to_show = list(df.columns[:20])
        sample = df[cols_to_show].head(MAX_DATASET_ROWS).to_string()

        context = f"""
        The user has uploaded a dataset with the following details:
        - Shape         : {df.shape[0]} rows and {df.shape[1]} columns
        - Column names  : {list(df.columns)}
        - Data types    : {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        - Sample rows   :
        {sample}
        """
        return context.strip()

    except Exception as e:
        logger.warning(f"[Chatbot] Could not build dataset context: {e}")
        return f"A dataset is loaded with {df.shape[0]} rows and {df.shape[1]} columns."


def _trim_chat_history(chat_history: list) -> list:
    """
    Keeps only the last MAX_HISTORY_TURNS full turns (user + assistant pairs).
    Each turn = 2 messages, so we keep last MAX_HISTORY_TURNS * 2 messages.
    This prevents the context window from filling up in long conversations.
    """
    max_messages = MAX_HISTORY_TURNS * 2
    if len(chat_history) > max_messages:
        logger.info(f"[Chatbot] Trimming chat history from {len(chat_history)} to {max_messages} messages.")
        return chat_history[-max_messages:]
    return chat_history


def _validate_reply(reply: str) -> str:
    """
    Validates the LLM reply before returning it to the user.
    Catches empty, too-short, or obviously truncated responses.
    """
    if not reply or not isinstance(reply, str):
        return "I'm sorry, I wasn't able to generate a response. Please try again."

    reply = reply.strip()

    if len(reply) < 5:
        return "I'm sorry, I wasn't able to generate a complete response. Please try rephrasing your question."

    return reply


def run_chatbot_agent(
    user_message : str,
    df           : pd.DataFrame = None,
    chat_history : list = []
) -> str:
    """
    Responds to the user's message in plain English.

    Args:
        user_message : what the user typed in the chat
        df           : the current dataset (optional)
        chat_history : list of previous messages {"role": ..., "content": ...}

    Returns:
        AI's reply as a clean, complete string
    """
    client = get_llm_client()

    # -------------------------------------------------------
    # STEP 1 — Build dataset context (size-limited)
    # -------------------------------------------------------
    dataset_context = _build_dataset_context(df)

    # -------------------------------------------------------
    # STEP 2 — System prompt defines the chatbot's personality
    # -------------------------------------------------------
    system_prompt = f"""
You are a friendly and concise data assistant.

Your job:
- Answer questions about the user's dataset simply
- Explain data science concepts in plain English
- Guide users on what to do next

Current dataset:
{dataset_context}

STRICT RULES:
- Keep ALL answers to 3 sentences maximum
- Never show tables, statistics, or numbers unless directly asked
- Never perform analysis — just answer the question conversationally
- If the user wants charts or deep analysis, tell them to use 
  the Analyse button instead
- Be warm, friendly and direct
- No bullet points unless the user specifically asks for a list
""".strip()

    # -------------------------------------------------------
    # STEP 3 — Build message list with trimmed history
    # -------------------------------------------------------
    messages = [{"role": "system", "content": system_prompt}]

    # Trim history to avoid context overflow
    trimmed_history = _trim_chat_history(chat_history)
    messages.extend(trimmed_history)

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    # -------------------------------------------------------
    # STEP 4 — Call the Groq API with max_tokens set
    # max_tokens prevents the model from hitting its default
    # limit mid-sentence, which caused the garbled output bug.
    # -------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_tokens=MAX_TOKENS    # FIX: this was missing before — caused truncation
        )
        reply = response.choices[0].message.content

    except Exception as e:
        logger.error(f"[Chatbot] Groq API call failed: {e}", exc_info=True)
        return (
            "I'm having trouble connecting to the AI service right now. "
            "Please check your API key and internet connection, then try again."
        )

    # -------------------------------------------------------
    # STEP 5 — Validate and return the reply
    # -------------------------------------------------------
    return _validate_reply(reply)


# -------------------------------------------------------
# Test block — simulates a multi-turn conversation
# -------------------------------------------------------
if __name__ == "__main__":
    print("Testing Chatbot Agent")
    print("=" * 40)

    test_data = {
        "name"  : ["Alice", "Bob", "Charlie"],
        "age"   : [25, 30, 27],
        "salary": [50000, 60000, 55000]
    }
    df = pd.DataFrame(test_data)

    chat_history = []

    questions = [
        "How many rows does my dataset have?",
        "What does correlation mean in simple terms?",
        "Which column should I clean first?"
    ]

    for question in questions:
        print(f"\nUser   : {question}")
        reply = run_chatbot_agent(
            user_message=question,
            df=df,
            chat_history=chat_history
        )
        print(f"Chatbot: {reply}")
        print("-" * 40)

        # Add turn to history
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant",  "content": reply})

    # Edge case: very long conversation (history trimming test)
    print("\n--- Testing history trimming (25 turns) ---")
    long_history = []
    for i in range(25):
        long_history.append({"role": "user",      "content": f"Question {i}"})
        long_history.append({"role": "assistant",  "content": f"Answer {i}"})

    reply = run_chatbot_agent(
        user_message="What is the mean of my age column?",
        df=df,
        chat_history=long_history
    )
    print(f"Chatbot: {reply}")