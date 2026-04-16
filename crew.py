# crew.py
# Central orchestrator of the Multi-Agent Data Intelligence System.
# Connects all 5 agents with robust routing, error handling, and validation.

import sys
import os
import logging
sys.path.append(os.getcwd())

import pandas as pd

# Import all agents
from agents.supervisor           import run_supervisor
from agents.cleaning_agent       import run_cleaning_agent
from agents.preprocessing_agent  import run_preprocessing_agent
from agents.analysis_agent       import run_analysis_agent
from agents.chatbot_agent        import run_chatbot_agent

# -------------------------------------------------------
# LOGGING SETUP
# Writes logs to both terminal and a log file for debugging
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crew_pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# ROUTING MAP
# Maps supervisor keywords → agent names.
# Far more robust than inline string matching.
# Add new keywords here without touching the routing logic.
# -------------------------------------------------------
ROUTING_MAP = {
    "cleaning"      : "cleaning",
    "clean"         : "cleaning",
    "remove"        : "cleaning",
    "duplicate"     : "cleaning",
    "missing"       : "cleaning",
    "preprocessing" : "preprocessing",
    "preprocess"    : "preprocessing",
    "encode"        : "preprocessing",
    "normalize"     : "preprocessing",
    "scale"         : "preprocessing",
    "analysis"      : "analysis",
    "analyse"       : "analysis",
    "analyze"       : "analysis",
    "visuali"       : "analysis",   # covers visualise / visualize / visualization
    "chatbot"       : "chatbot",
    "chat"          : "chatbot",
}


def _resolve_agent(decision: str) -> str:
    """
    Converts the supervisor's raw decision string into a clean agent name.
    Tries exact match first, then keyword scan, then defaults to chatbot.
    """
    decision_lower = decision.strip().lower()

    # Exact match (e.g. supervisor returns exactly "cleaning")
    if decision_lower in ROUTING_MAP:
        return ROUTING_MAP[decision_lower]

    # Keyword scan (e.g. supervisor returns "run data_cleaning now")
    for keyword, agent in ROUTING_MAP.items():
        if keyword in decision_lower:
            return agent

    # Nothing matched — default to chatbot
    logger.warning(f"[Crew] Could not resolve decision '{decision}' — defaulting to chatbot.")
    return "chatbot"


def _validate_response_text(text: str, agent_name: str) -> str:
    """
    Guards against garbled, truncated, or empty LLM responses.
    Returns the original text if clean, or a safe fallback message.
    """
    if not text or not isinstance(text, str):
        logger.error(f"[Crew] {agent_name} returned empty or non-string response.")
        return f"The {agent_name} agent completed the task but could not generate an explanation. Please try again."

    # Strip and check minimum length
    text = text.strip()
    if len(text) < 10:
        logger.warning(f"[Crew] {agent_name} response is suspiciously short: '{text}'")
        return f"The {agent_name} agent completed the task but returned an incomplete response. Please try again."

    # Check for common garbled endings (repeated characters, broken words)
    last_50 = text[-50:]
    garble_signals = ["thisis", "ithe", "tthis", "net output", "  "]
    for signal in garble_signals:
        if signal in last_50.lower():
            logger.warning(f"[Crew] Garbled text detected in {agent_name} response. Trimming.")
            # Trim at last clean sentence ending
            for end_char in [". ", ".\n", "! ", "? "]:
                last_clean = text.rfind(end_char)
                if last_clean != -1:
                    return text[:last_clean + 1].strip()

    return text


def _validate_dataframe(df: pd.DataFrame, agent_name: str) -> dict | None:
    """
    Validates the dataframe before passing it to an agent.
    Returns an error dict if invalid, None if all is fine.
    """
    if df is None:
        return {
            "agent"  : agent_name,
            "success": False,
            "message": "Please upload a dataset first before running this operation."
        }

    if not isinstance(df, pd.DataFrame):
        return {
            "agent"  : agent_name,
            "success": False,
            "message": "The uploaded data is not a valid dataset. Please upload a CSV or Excel file."
        }

    if df.empty:
        return {
            "agent"  : agent_name,
            "success": False,
            "message": "Your dataset appears to be empty. Please upload a file with data."
        }

    if df.shape[1] == 0:
        return {
            "agent"  : agent_name,
            "success": False,
            "message": "Your dataset has no columns. Please check your file and re-upload."
        }

    return None  # All good


# -------------------------------------------------------
# AGENT RUNNERS (with error handling)
# Each function wraps an agent call in try/except so a
# single agent failure never crashes the whole app.
# -------------------------------------------------------

def _run_cleaning(df: pd.DataFrame) -> dict:
    validation_error = _validate_dataframe(df, "cleaning")
    if validation_error:
        return validation_error
    try:
        logger.info("[Crew] Running Cleaning Agent...")
        cleaned_df, summary = run_cleaning_agent(df)
        summary = _validate_response_text(summary, "cleaning")
        logger.info(f"[Crew] Cleaning complete. Shape: {df.shape} → {cleaned_df.shape}")
        return {
            "agent"  : "cleaning",
            "success": True,
            "df"     : cleaned_df,
            "message": summary
        }
    except Exception as e:
        logger.error(f"[Crew] Cleaning Agent crashed: {e}", exc_info=True)
        return {
            "agent"  : "cleaning",
            "success": False,
            "message": f"The Cleaning Agent encountered an error: {str(e)}. Please try again or check your dataset."
        }


def _run_preprocessing(df: pd.DataFrame) -> dict:
    validation_error = _validate_dataframe(df, "preprocessing")
    if validation_error:
        return validation_error
    try:
        logger.info("[Crew] Running Preprocessing Agent...")
        processed_df, summary = run_preprocessing_agent(df)
        summary = _validate_response_text(summary, "preprocessing")
        logger.info(f"[Crew] Preprocessing complete. Shape: {df.shape} → {processed_df.shape}")
        return {
            "agent"  : "preprocessing",
            "success": True,
            "df"     : processed_df,
            "message": summary
        }
    except Exception as e:
        logger.error(f"[Crew] Preprocessing Agent crashed: {e}", exc_info=True)
        return {
            "agent"  : "preprocessing",
            "success": False,
            "message": f"The Preprocessing Agent encountered an error: {str(e)}. Please try again or check your dataset."
        }


def _run_analysis(df: pd.DataFrame) -> dict:
    validation_error = _validate_dataframe(df, "analysis")
    if validation_error:
        return validation_error
    try:
        logger.info("[Crew] Running Analysis Agent...")
        stats, fig, summary = run_analysis_agent(df)
        summary = _validate_response_text(summary, "analysis")
        logger.info("[Crew] Analysis complete.")
        return {
            "agent"  : "analysis",
            "success": True,
            "stats"  : stats,
            "fig"    : fig,
            "message": summary
        }
    except Exception as e:
        logger.error(f"[Crew] Analysis Agent crashed: {e}", exc_info=True)
        return {
            "agent"  : "analysis",
            "success": False,
            "message": f"The Analysis Agent encountered an error: {str(e)}. Please try again or check your dataset."
        }


def _run_chatbot(user_request: str, df: pd.DataFrame, chat_history: list) -> dict:
    try:
        logger.info("[Crew] Running Chatbot Agent...")
        reply = run_chatbot_agent(
            user_message=user_request,
            df=df,
            chat_history=chat_history
        )
        reply = _validate_response_text(reply, "chatbot")
        logger.info("[Crew] Chatbot response generated.")
        return {
            "agent"  : "chatbot",
            "success": True,
            "message": reply,
            "fig"    : None,
            "stats"  : None
        }
    except Exception as e:
        logger.error(f"[Crew] Chatbot Agent crashed: {e}", exc_info=True)
        return {
            "agent"  : "chatbot",
            "success": False,
            "message": f"The Chatbot encountered an error: {str(e)}. Please try again.",
            "fig"    : None,
            "stats"  : None
        }


# -------------------------------------------------------
# MAIN CREW FUNCTION
# -------------------------------------------------------
def run_crew(
    user_request : str,
    df           : pd.DataFrame = None,
    chat_history : list = []
) -> dict:
    """
    Main function that runs the entire multi-agent pipeline.

    Args:
        user_request : what the user typed or clicked
        df           : the current dataset (if uploaded)
        chat_history : conversation history for the chatbot

    Returns:
        dict with keys depending on which agent ran:
          - agent    : name of the agent that ran
          - success  : True/False
          - message  : human-readable explanation
          - df       : (cleaning/preprocessing only) processed dataframe
          - stats    : (analysis only) statistics text
          - fig      : (analysis only) plotly figure
    """
    logger.info(f"[Crew] ── New Request ──────────────────────────────")
    logger.info(f"[Crew] User request: '{user_request}'")

    # Guard against empty requests
    if not user_request or not user_request.strip():
        return {
            "agent"  : "chatbot",
            "success": False,
            "message": "Please type a message or click an action button."
        }

    # STEP 1 — Supervisor decides which agent to use
    try:
        raw_decision = run_supervisor(user_request)
        logger.info(f"[Crew] Supervisor raw decision: '{raw_decision}'")
    except Exception as e:
        logger.error(f"[Crew] Supervisor crashed: {e}", exc_info=True)
        # Supervisor down — fall back to chatbot
        return _run_chatbot(user_request, df, chat_history)

    # STEP 2 — Resolve the decision to a clean agent name
    agent_name = _resolve_agent(raw_decision)
    logger.info(f"[Crew] Resolved agent: '{agent_name}'")

    # STEP 3 — Route to the correct agent
    AGENT_RUNNERS = {
        "cleaning"      : lambda: _run_cleaning(df),
        "preprocessing" : lambda: _run_preprocessing(df),
        "analysis"      : lambda: _run_analysis(df),
        "chatbot"       : lambda: _run_chatbot(user_request, df, chat_history),
    }

    runner = AGENT_RUNNERS.get(agent_name, lambda: _run_chatbot(user_request, df, chat_history))
    result = runner()

    logger.info(f"[Crew] Result — agent={result['agent']}, success={result['success']}")
    return result


# -------------------------------------------------------
# Test block — tests the full pipeline end to end
# -------------------------------------------------------
if __name__ == "__main__":
    print("Testing Full Crew Pipeline")
    print("=" * 40)

    test_data = {
        "name"  : ["Alice", "Bob", "Charlie", "Bob", None],
        "age"   : [25, 30, None, 30, 22],
        "salary": [50000, 60000, 55000, 60000, None]
    }
    df = pd.DataFrame(test_data)

    print("\n--- Test 1: Cleaning Request ---")
    result = run_crew("remove duplicates and fix missing values", df)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")
    if result['success']:
        print(f"Cleaned DataFrame shape: {result['df'].shape}")

    print("\n--- Test 2: Analysis Request ---")
    result = run_crew("show me statistics of my data", df)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")

    print("\n--- Test 3: Chatbot Request ---")
    result = run_crew("what is standard deviation?", df)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")

    print("\n--- Test 4: Empty Request (edge case) ---")
    result = run_crew("", df)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")

    print("\n--- Test 5: No Dataset (edge case) ---")
    result = run_crew("clean my data", None)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")

    print("\n--- Test 6: Ambiguous keyword (edge case) ---")
    result = run_crew("run data_cleaning now", df)
    print(f"Agent   : {result['agent']}")
    print(f"Success : {result['success']}")