# agents/supervisor.py
# Reads the user's request and decides which agent should handle it.
# Returns exactly one word: cleaning / preprocessing / analysis / chatbot

import sys
import os
import logging
sys.path.append(os.getcwd())

from utils.llm_config import get_llm_client, MODEL_NAME

logger = logging.getLogger(__name__)

# Valid agent names — used to validate the supervisor's response
VALID_AGENTS = {"cleaning", "preprocessing", "analysis", "chatbot"}


def run_supervisor(user_request: str) -> str:
    """
    Takes the user's request and returns which agent should handle it.

    Args:
        user_request : the raw message from the user

    Returns:
        One of: "cleaning", "preprocessing", "analysis", "chatbot"
        Falls back to "chatbot" if the response is unexpected.
    """
    client = get_llm_client()

    system_prompt = system_prompt = """
You are a Supervisor Agent for a data intelligence system.
Your ONLY job is to read the user's request and return ONE word.

Rules:
- Return "cleaning" ONLY if user explicitly asks to clean, 
  remove duplicates, fix missing values, or handle outliers
- Return "preprocessing" ONLY if user explicitly asks to 
  encode, normalize, scale, or transform data
- Return "analysis" ONLY if user explicitly asks for a chart,
  graph, visualization, correlation, or plot
- Return "chatbot" for EVERYTHING else including:
  questions about the data, explanations, how many rows,
  what columns exist, what does X mean, tell me about,
  describe, summarize, statistics questions in plain English

When in doubt → always return "chatbot"

Reply with ONLY one word. No punctuation. No explanation.

Examples:
"remove duplicates" → cleaning
"normalize the data" → preprocessing
"show me a correlation chart" → analysis
"show me a bar graph" → analysis
"how many rows do I have?" → chatbot
"what columns are there?" → chatbot
"tell me about my data" → chatbot
"what is the average age?" → chatbot
"explain standard deviation" → chatbot
"what does correlation mean?" → chatbot
"give me insights" → chatbot
"analyse my data" → analysis
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_request}
            ],
            temperature=0.1,     # Very low — we want deterministic routing
            max_tokens=10        # We only need one word — no need for more tokens
        )

        decision = response.choices[0].message.content.strip().lower()

        # Remove any accidental punctuation the LLM might add
        decision = decision.strip(".,!?\"'")

        # Validate — if the LLM returned something unexpected, default to chatbot
        if decision not in VALID_AGENTS:
            logger.warning(
                f"[Supervisor] Unexpected decision '{decision}' — defaulting to chatbot."
            )
            return "chatbot"

        logger.info(f"[Supervisor] Routed '{user_request[:50]}' → '{decision}'")
        return decision

    except Exception as e:
        logger.error(f"[Supervisor] API call failed: {e}", exc_info=True)
        # If supervisor crashes, default to chatbot so the user gets some response
        return "chatbot"


# -------------------------------------------------------
# Test block
# -------------------------------------------------------
if __name__ == "__main__":
    test_requests = [
        "remove all duplicate rows from my dataset",
        "normalize the salary column",
        "show me a bar chart of the data",
        "what does correlation mean?",
        "fix missing values in age column",
        "encode my gender column",
        "what is the standard deviation?",
        "run DATA_CLEANING NOW",          # edge case: unexpected casing/format
    ]

    print("Testing Supervisor Agent")
    print("=" * 40)

    for request in test_requests:
        result = run_supervisor(request)
        print(f"Request : {request}")
        print(f"Decision: {result}")
        print("-" * 40)