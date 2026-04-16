# agents/analysis_agent.py
# This agent analyses a DataFrame and produces statistics,
# correlation info, and a visual chart using Plotly.

import sys
import os
import logging
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import plotly.express as px
from utils.llm_config import get_llm_client, MODEL_NAME

logger = logging.getLogger(__name__)

# FIX: Create the outputs folder automatically if it doesn't exist.
# This was the cause of the FileNotFoundError crash.
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def run_analysis_agent(df: pd.DataFrame) -> tuple:
    """
    Analyses the given DataFrame: statistics, correlations, and a chart.

    Args:
        df : pandas DataFrame (cleaned or preprocessed)

    Returns:
        (stats_text, fig, summary_text)
        stats_text   = raw statistics as a readable string
        fig          = a Plotly correlation heatmap figure (or None)
        summary_text = plain English AI explanation of findings
    """

    # -------------------------------------------------------
    # STEP 1 — Basic statistics for every numeric column
    # describe() gives: count, mean, std, min, 25%, 50%, 75%, max
    # -------------------------------------------------------
    try:
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            logger.warning("[Analysis] No numeric columns found in dataset.")
            return (
                "No numeric columns found.",
                None,
                "Your dataset doesn't have any numeric columns to analyse. "
                "Try uploading a dataset with numbers such as age, salary, or scores."
            )

        stats      = numeric_df.describe()
        stats_text = stats.to_string()
        logger.info(f"[Analysis] Computed statistics for {len(numeric_df.columns)} numeric column(s).")

    except Exception as e:
        logger.error(f"[Analysis] Failed to compute statistics: {e}", exc_info=True)
        return (
            "Could not compute statistics.",
            None,
            f"The Analysis Agent encountered an error while computing statistics: {e}"
        )

    # -------------------------------------------------------
    # STEP 2 — Correlation matrix
    # Correlation shows how strongly two columns are related.
    #   1.0  = perfectly positively correlated
    #  -1.0  = perfectly negatively correlated
    #   0.0  = no relationship
    # We skip correlation if there's only 1 numeric column
    # (correlation needs at least 2 columns to be meaningful).
    # -------------------------------------------------------
    correlation_text = "Not enough numeric columns for correlation."
    correlation      = None

    try:
        if len(numeric_df.columns) >= 2:
            correlation      = numeric_df.corr().round(2)
            correlation_text = correlation.to_string()
            logger.info("[Analysis] Computed correlation matrix.")
        else:
            logger.info("[Analysis] Skipping correlation — only 1 numeric column.")

    except Exception as e:
        logger.warning(f"[Analysis] Could not compute correlation: {e}")
        correlation_text = f"Correlation could not be computed: {e}"

    # -------------------------------------------------------
    # STEP 3 — Create a correlation heatmap chart
    # Dark red = strong positive, Dark blue = strong negative
    # We skip this if correlation couldn't be computed.
    # -------------------------------------------------------
    fig = None

    try:
        if correlation is not None:
            fig = px.imshow(
                correlation,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=True
            )
            fig.update_layout(
                title_font_size=16,
                width=600,
                height=500
            )
            logger.info("[Analysis] Correlation heatmap created.")

    except Exception as e:
        logger.warning(f"[Analysis] Could not create chart: {e}")
        fig = None

    # -------------------------------------------------------
    # STEP 4 — Ask AI to explain findings in plain English
    # -------------------------------------------------------
    analysis_report = (
        f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        f"Numeric columns: {list(numeric_df.columns)}\n\n"
        f"Basic Statistics:\n{stats_text}\n\n"
        f"Correlation Matrix:\n{correlation_text}"
    )

    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data analysis assistant. "
                        "You will receive statistics and correlation data from a dataset. "
                        "Explain the most interesting findings clearly to a non-technical user. "
                        "Keep it to 5 to 7 complete sentences — never cut off mid-word. "
                        "Start with: 'Here is what I found in your dataset.' "
                        "Mention specific numbers to make the explanation concrete."
                    )
                },
                {
                    "role": "user",
                    "content": analysis_report
                }
            ],
            temperature=0.3,
            max_tokens=400    # FIX: was missing — caused truncated responses
        )

        summary_text = response.choices[0].message.content.strip()

        if not summary_text:
            summary_text = (
                "Here is what I found in your dataset.\n\n"
                f"Statistics:\n{stats_text}"
            )

    except Exception as e:
        logger.error(f"[Analysis] LLM summary failed: {e}", exc_info=True)
        summary_text = (
            f"Here is what I found in your dataset.\n\n"
            f"Statistics:\n{stats_text}\n\n"
            f"Correlation:\n{correlation_text}"
        )

    return stats_text, fig, summary_text


# -------------------------------------------------------
# Test block
# -------------------------------------------------------
if __name__ == "__main__":
    print("Testing Analysis Agent")
    print("=" * 40)

    test_data = {
        "age"       : [25, 30, 27, 22, 35, 28, 40, 33],
        "salary"    : [50000, 60000, 55000, 45000, 70000, 58000, 80000, 65000],
        "experience": [2, 5, 3, 1, 8, 4, 12, 7]
    }
    df = pd.DataFrame(test_data)

    print("Dataset:")
    print(df)
    print()

    stats, fig, summary = run_analysis_agent(df)

    print("Statistics:")
    print(stats)
    print()
    print("AI Summary:")
    print(summary)
    print()

    # Save the chart — now safe because outputs/ is created automatically above
    if fig:
        chart_path = os.path.join(OUTPUTS_DIR, "correlation_chart.html")
        fig.write_html(chart_path)
        print(f"Chart saved to {chart_path}")
        print("Open that file in your browser to see the chart!")

    # Edge case: only 1 numeric column (correlation should be skipped gracefully)
    print("\n--- Edge case: Single numeric column ---")
    single_col_df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age" : [25, 30, 27]
    })
    stats2, fig2, summary2 = run_analysis_agent(single_col_df)
    print(f"Fig returned: {fig2}")
    print(f"Summary: {summary2}")

    # Edge case: no numeric columns at all
    print("\n--- Edge case: No numeric columns ---")
    text_only_df = pd.DataFrame({
        "name"  : ["Alice", "Bob"],
        "city"  : ["London", "Paris"]
    })
    stats3, fig3, summary3 = run_analysis_agent(text_only_df)
    print(f"Summary: {summary3}")