import sys
import os
import logging

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.llm_config import get_llm_client, MODEL_NAME

logger = logging.getLogger(__name__)

MAX_UNIQUE_FOR_ENCODING = 20


def run_preprocessing_agent(df: pd.DataFrame) -> tuple:
    changes = []
    df = df.copy()

    # -------------------------------------------------------
    # STEP 1 — Detect text / categorical columns
    # -------------------------------------------------------
    text_columns = []
    for col in df.columns:
        if (
            not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_datetime64_any_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        ):
            # Skip date columns first — they're expanded in Step 1.5
            if "date" in col.lower():
                continue

            # Skip likely ID columns
            if "id" in col.lower():
                continue

            unique_ratio = df[col].nunique() / max(len(df), 1)

            # FIX: threshold raised from 0.5 → 0.8.
            # 0.5 was too aggressive for small datasets: a legitimate
            # categorical column (e.g. City with 4 cities across 7 rows)
            # has ratio ≈ 0.57 and was silently skipped.  At 0.8, only
            # near-unique columns like Name (ratio ≈ 1.0) are skipped,
            # while real categoricals like City/Gender/Status pass through.
            if unique_ratio > 0.8:
                changes.append(f"Skipped '{col}' (high cardinality, ratio={unique_ratio:.2f})")
                continue

            text_columns.append(col)

    # -------------------------------------------------------
    # STEP 1.5 — Parse & expand date columns
    # -------------------------------------------------------
    for col in list(df.columns):          # iterate a snapshot so drops are safe
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col + "_year"]  = df[col].dt.year
                df[col + "_month"] = df[col].dt.month
                df[col + "_day"]   = df[col].dt.day
                df.drop(columns=[col], inplace=True)
                changes.append(f"Converted '{col}' to datetime features (_year, _month, _day)")
                logger.info(f"[Preprocessing] Expanded date column '{col}'")
            except Exception as e:
                logger.warning(f"[Preprocessing] Date handling failed for '{col}': {e}")

    # FIX — prune any text_columns entries that were just dropped
    text_columns = [c for c in text_columns if c in df.columns]

    # -------------------------------------------------------
    # STEP 1.6 — Impute missing values in categorical columns
    #            (must happen BEFORE LabelEncoder so NaN is
    #             never encoded as the string "nan")
    # -------------------------------------------------------
    for col in text_columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)          # pandas-3.0 safe (no inplace)
            changes.append(f"Filled missing values in '{col}' with mode='{fill_val}'")
            logger.info(f"[Preprocessing] Imputed NaN in categorical column '{col}' → '{fill_val}'")

    # -------------------------------------------------------
    # STEP 2 — Encode categorical columns with LabelEncoder
    # -------------------------------------------------------
    label_encoder = LabelEncoder()
    encoded_col_names = []          # track which cols were encoded (not scaled)

    for col in text_columns:
        if col not in df.columns:   # safety guard
            continue
        try:
            unique_values = df[col].nunique()
            if unique_values <= MAX_UNIQUE_FOR_ENCODING:
                original_values = sorted(df[col].dropna().unique().tolist())
                df[col] = label_encoder.fit_transform(df[col].astype(str))
                encoded_col_names.append(col)
                changes.append(f"Encoded '{col}': {original_values} → integers")
                logger.info(f"[Preprocessing] Label-encoded column '{col}'")
            else:
                changes.append(
                    f"Skipped encoding '{col}' (too many unique values: {unique_values})"
                )
        except Exception as e:
            logger.warning(f"[Preprocessing] Could not encode column '{col}': {e}")
            changes.append(f"Could not encode '{col}' — skipped.")

    # -------------------------------------------------------
    # STEP 2.5 — Impute missing values in numeric columns
    #            using the COLUMN MEAN before scaling.
    #            MinMaxScaler raises ValueError on NaN, so
    #            this step is mandatory for correct results.
    # -------------------------------------------------------
    numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in numeric_cols_raw if c not in encoded_col_names]

    for col in cols_to_impute:
        if df[col].isnull().any():
            col_mean = df[col].mean()           # correct mean (ignores NaN by default)
            df[col] = df[col].fillna(col_mean)  # pandas-3.0 safe (no inplace)
            changes.append(
                f"Filled {df[col].isnull().sum()} missing value(s) in '{col}' "
                f"with mean={col_mean:.4f}"
            )
            logger.info(f"[Preprocessing] Imputed NaN in numeric column '{col}' with mean={col_mean:.4f}")

    # -------------------------------------------------------
    # STEP 3 — Scale numeric columns with MinMaxScaler
    #          Encoded categorical columns are EXCLUDED so
    #          their integer labels are not compressed to [0,1].
    #
    # FIX: ID columns are also excluded from scaling.
    #      Previously CustomerID (and any *_id column) was
    #      passed to MinMaxScaler alongside Age/Salary, which
    #      squashed the identifier to [0,1] — destroying its
    #      identity and making it useless for record lookup.
    # -------------------------------------------------------
    id_col_names    = [c for c in df.columns if "id" in c.lower()]
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [
        c for c in numeric_columns
        if c not in encoded_col_names and c not in id_col_names
    ]

    if numeric_columns:
        try:
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            changes.append(
                f"MinMax-scaled {len(numeric_columns)} numeric column(s): {numeric_columns}"
            )
            logger.info(f"[Preprocessing] Scaled columns: {numeric_columns}")
        except Exception as e:
            logger.warning(f"[Preprocessing] Scaling failed: {e}")
            changes.append(f"Scaling could not be completed: {e}")
    else:
        changes.append("No numeric columns found to scale.")
        logger.info("[Preprocessing] No numeric columns to scale.")

    # -------------------------------------------------------
    # STEP 4 — Build report and request AI summary
    # -------------------------------------------------------
    changes_text = (
        "\n".join(f"- {c}" for c in changes)
        if changes
        else "No preprocessing was needed."
    )

    preprocessing_report = (
        f"Preprocessing completed on dataset with "
        f"{len(df)} rows and {len(df.columns)} columns.\n\n"
        f"Steps performed:\n{changes_text}"
    )

    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data preprocessing assistant. "
                        "Explain clearly in 4–6 sentences. "
                        "Start with: 'I have finished preprocessing your dataset.'"
                    ),
                },
                {
                    "role": "user",
                    "content": preprocessing_report,
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary_text = response.choices[0].message.content.strip()
        if not summary_text:
            summary_text = (
                "I have finished preprocessing your dataset.\n"
                f"{changes_text}"
            )
    except Exception as e:
        logger.error(f"[Preprocessing] LLM summary failed: {e}", exc_info=True)
        summary_text = (
            f"I have finished preprocessing your dataset.\n\n{changes_text}"
        )

    return df, summary_text
