import sys
import os
import logging

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

from utils.llm_config import get_llm_client, MODEL_NAME

logger = logging.getLogger(__name__)


def run_cleaning_agent(df: pd.DataFrame) -> tuple:
    """
    Cleans the given DataFrame automatically.

    Args:
        df : the raw pandas DataFrame uploaded by the user

    Returns:
        (cleaned_df, summary_text)
        cleaned_df   = the cleaned version of the data
        summary_text = plain English explanation of what was done
    """
    # FIX — guard against empty DataFrame before touching anything
    if df is None or df.empty:
        logger.warning("[Cleaning] Received empty DataFrame — nothing to clean.")
        return df if df is not None else pd.DataFrame(), \
               "I have finished cleaning your dataset. The dataset appears to be empty, so no changes were needed."

    # Work on a copy — never modify the caller's original DataFrame
    df = df.copy()
    original_rows = len(df)
    original_cols = len(df.columns)
    changes = []

    # -------------------------------------------------------
    # STEP 1 — Drop completely empty rows and columns
    # These carry zero information and cause downstream issues.
    # FIX: use assignment form (df = df.dropna(...)) instead of
    #      inplace=True, which is deprecated in pandas 3.0.
    # -------------------------------------------------------
    try:
        rows_before = len(df)
        df = df.dropna(how="all")                       # rows where every cell is NaN
        empty_rows_dropped = rows_before - len(df)
        if empty_rows_dropped > 0:
            changes.append(f"Removed {empty_rows_dropped} completely empty row(s)")
            logger.info(f"[Cleaning] Dropped {empty_rows_dropped} fully-empty rows")

        cols_before = len(df.columns)
        df = df.dropna(axis=1, how="all")               # columns where every value is NaN
        empty_cols_dropped = cols_before - len(df.columns)
        if empty_cols_dropped > 0:
            changes.append(f"Removed {empty_cols_dropped} completely empty column(s)")
            logger.info(f"[Cleaning] Dropped {empty_cols_dropped} fully-empty columns")
    except Exception as e:
        logger.warning(f"[Cleaning] Could not drop empty rows/cols: {e}")

    # -------------------------------------------------------
    # STEP 2 — Remove duplicate rows
    # keep='first' preserves the first occurrence.
    #
    # FIX: comparing ALL columns (including CustomerID / auto-
    # increment IDs) misses content-identical rows that happen
    # to have different ID values (e.g. Alice row 1 vs row 8).
    # We detect duplicates on CONTENT columns only — any column
    # whose name contains "id" is excluded from the comparison.
    # -------------------------------------------------------
    try:
        id_cols    = [c for c in df.columns if "id" in c.lower()]
        subset     = [c for c in df.columns if c not in id_cols] or None
        dups_found = df.duplicated(subset=subset).sum()
        if dups_found > 0:
            df = df.drop_duplicates(subset=subset, keep="first")
            changes.append(f"Removed {dups_found} duplicate row(s) (content match, ID columns excluded)")
            logger.info(f"[Cleaning] Removed {dups_found} content-duplicates")
    except Exception as e:
        logger.warning(f"[Cleaning] Could not remove duplicates: {e}")

    # -------------------------------------------------------
    # STEP 3 — Handle missing values column by column
    # Strategy:
    #   Numeric columns → fill with column MEDIAN
    #                     (robust to skew, unlike mean)
    #   Date columns    → leave as NaN so the preprocessing
    #                     agent can parse and expand them
    #                     properly (datetime features).
    #   Low-cardinality text (≤ 10 unique non-null values,
    #     e.g. Gender, City) → fill with column MODE.
    #
    #     FIX: previously ALL text NaN were filled with the
    #     string "Unknown". This poisoned binary columns:
    #     Gender [Male, Female] became [Male, Female, Unknown]
    #     and LabelEncoder produced 0/1/2 instead of 0/1.
    #     It also inflated City's unique-value count, pushing
    #     its cardinality ratio above the preprocessing skip
    #     threshold, causing City to be silently dropped from
    #     encoding entirely.
    #
    #   High-cardinality text (> 10 unique non-null values,
    #     e.g. free-text fields) → fill with "Unknown" as
    #     before, since no single mode is meaningful.
    #
    # FIX: guard against median() returning NaN when the
    #      entire column is missing.
    # -------------------------------------------------------
    for column in df.columns:
        try:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()

                # Guard: median is NaN when ALL values are NaN
                if pd.isna(median_value):
                    logger.warning(
                        f"[Cleaning] Median for '{column}' is NaN "
                        f"(all values missing) — filling with 0"
                    )
                    median_value = 0.0
                    changes.append(
                        f"Filled {missing_count} missing value(s) in "
                        f"'{column}' with 0 (all values were missing)"
                    )
                else:
                    changes.append(
                        f"Filled {missing_count} missing value(s) in "
                        f"'{column}' with median ({median_value:.2f})"
                    )

                df[column] = df[column].fillna(median_value)
                logger.info(f"[Cleaning] Filled '{column}' NaNs with median {median_value:.2f}")

            elif "date" in column.lower():
                # Leave date NaN intact — preprocessing agent converts the
                # column to datetime and handles NaN with mean imputation
                # on the resulting _year/_month/_day features.
                changes.append(
                    f"Left {missing_count} missing value(s) in date column "
                    f"'{column}' for preprocessing agent to handle"
                )
                logger.info(f"[Cleaning] Skipped NaN fill for date column '{column}'")

            else:
                # Text column: choose fill strategy based on cardinality
                n_unique = df[column].nunique()   # ignores NaN by default
                if n_unique <= 10:
                    # Low-cardinality (e.g. Gender, City, Status) → use mode
                    mode_series = df[column].mode()
                    fill_value  = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    df[column]  = df[column].fillna(fill_value)
                    changes.append(
                        f"Filled {missing_count} missing value(s) in "
                        f"'{column}' with mode ('{fill_value}')"
                    )
                    logger.info(f"[Cleaning] Filled '{column}' NaNs with mode '{fill_value}'")
                else:
                    # High-cardinality (e.g. free-text descriptions) → "Unknown"
                    df[column] = df[column].fillna("Unknown")
                    changes.append(
                        f"Filled {missing_count} missing value(s) in "
                        f"'{column}' with 'Unknown'"
                    )
                    logger.info(f"[Cleaning] Filled '{column}' NaNs with 'Unknown'")

        except Exception as e:
            logger.warning(f"[Cleaning] Could not handle missing values in '{column}': {e}")

    # -------------------------------------------------------
    # STEP 4 — Remove outliers using the IQR method
    #
    # FIX: Previously, outlier masks were applied one column at
    # a time inside the loop, which caused CASCADING IQR DRIFT:
    # after removing row R for column A, column B's Q1/Q3 were
    # recomputed on the already-reduced dataset, shifting the
    # outlier boundaries and silently removing additional rows
    # that would NOT have been flagged on the original data.
    #
    # Correct approach:
    #   1. Compute each column's outlier mask on the ORIGINAL df
    #   2. Combine all masks: a row is removed only if it is an
    #      outlier in AT LEAST ONE numeric column
    #   3. Apply the combined mask exactly once
    #
    # This keeps all IQR calculations consistent and ensures
    # no row is removed as a side-effect of another column's
    # earlier removal.
    # -------------------------------------------------------
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Start with "keep everything" mask
    combined_keep_mask = pd.Series(True, index=df.index)
    outlier_details = []   # collect messages before we know if we'll actually remove

    for column in numeric_columns:
        try:
            Q1  = df[column].quantile(0.25)
            Q3  = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Skip constant columns — IQR of 0 means no outliers by definition
            if IQR == 0:
                logger.info(f"[Cleaning] Skipping outlier check for '{column}' — IQR is 0")
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_keep    = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            outlier_count = (~col_keep).sum()

            if outlier_count > 0:
                outlier_details.append((column, outlier_count, lower_bound, upper_bound, col_keep))
                combined_keep_mask &= col_keep

        except Exception as e:
            logger.warning(f"[Cleaning] Outlier check failed for '{column}': {e}")

    # Apply combined mask once — after we know the final row count
    if outlier_details:
        rows_after = combined_keep_mask.sum()
        if rows_after < 3:
            changes.append(
                "Skipped outlier removal — would leave fewer than 3 rows in the dataset"
            )
            logger.warning("[Cleaning] Outlier removal skipped — too few rows would remain")
        else:
            df = df[combined_keep_mask].copy()
            for (col, cnt, lb, ub, _) in outlier_details:
                changes.append(
                    f"Removed {cnt} outlier(s) from '{col}' "
                    f"(outside range [{lb:.2f}, {ub:.2f}])"
                )
                logger.info(f"[Cleaning] Removed {cnt} outliers from '{col}'")

    # -------------------------------------------------------
    # STEP 5 — Reset the index after all row removals
    # -------------------------------------------------------
    df = df.reset_index(drop=True)

    # -------------------------------------------------------
    # STEP 6 — Build the summary report
    # -------------------------------------------------------
    cleaned_rows = len(df)
    changes_text = (
        "\n".join(f"- {c}" for c in changes)
        if changes
        else "No issues found in the dataset."
    )

    cleaning_report = (
        f"Original dataset: {original_rows} rows, {original_cols} columns\n"
        f"Cleaned dataset : {cleaned_rows} rows, {len(df.columns)} columns\n"
        f"Rows removed    : {original_rows - cleaned_rows}\n\n"
        f"Changes made:\n{changes_text}"
    )

    # -------------------------------------------------------
    # STEP 7 — Ask the AI to explain the report in plain English
    # -------------------------------------------------------
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data cleaning assistant. "
                        "You will receive a technical cleaning report. "
                        "Explain it clearly and in a friendly way to a non-technical user. "
                        "Keep it short — 4 to 6 complete sentences maximum. "
                        "Always write complete sentences, never cut off mid-word. "
                        "Start with: 'I have finished cleaning your dataset.'"
                    ),
                },
                {
                    "role": "user",
                    "content": cleaning_report,
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary_text = response.choices[0].message.content.strip()
        if not summary_text:
            summary_text = f"I have finished cleaning your dataset.\n\n{changes_text}"
    except Exception as e:
        logger.error(f"[Cleaning] LLM summary failed: {e}", exc_info=True)
        summary_text = f"I have finished cleaning your dataset.\n\n{changes_text}"

    return df, summary_text


# -------------------------------------------------------
# Test block — runs only when this file is executed directly
# -------------------------------------------------------
if __name__ == "__main__":
    print("Testing Cleaning Agent")
    print("=" * 40)

    test_data = {
        "name":   ["Alice", "Bob", "Charlie", "Bob", None, "Eve"],
        "age":    [25, 30, None, 30, 22, 999],    # 999 is outlier; None is missing
        "salary": [50000, 60000, 55000, 60000, None, 52000],
    }
    df = pd.DataFrame(test_data)
    print("Original Data:")
    print(df)
    print()

    cleaned_df, summary = run_cleaning_agent(df)
    print("Cleaned Data:")
    print(cleaned_df)
    print()
    print("AI Summary:")
    print(summary)

    # Edge case: empty dataset
    print("\n--- Edge case: Empty DataFrame ---")
    empty_df = pd.DataFrame()
    result_df, result_summary = run_cleaning_agent(empty_df)
    print(result_summary)

    # Edge case: all-null column
    print("\n--- Edge case: All-null column ---")
    null_col_data = {
        "name":  ["Alice", "Bob", "Charlie"],
        "age":   [25, 30, 27],
        "empty": [None, None, None],
    }
    null_df = pd.DataFrame(null_col_data)
    result_df, result_summary = run_cleaning_agent(null_df)
    print(f"Columns after cleaning: {list(result_df.columns)}")
    print(result_summary)

    # Edge case: all-NaN numeric column (tests median NaN guard)
    print("\n--- Edge case: All-NaN numeric column ---")
    nan_num_data = {
        "name":   ["Alice", "Bob", "Charlie"],
        "salary": [np.nan, np.nan, np.nan],
    }
    nan_df = pd.DataFrame(nan_num_data)
    result_df, result_summary = run_cleaning_agent(nan_df)
    print(f"Salary column after cleaning: {result_df['salary'].tolist()}")
    assert result_df["salary"].isnull().sum() == 0, "FAIL: NaN not filled in all-NaN column"
    print("✅ All-NaN column guard works correctly.")
