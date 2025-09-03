import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import torch
import logging
import streamlit as st
import pandas as pd
from app.pdf_extractor import extract_text_from_pdf
from app.evaluation import evaluate_article
from app.config import MAX_TRANSITION_WORDS, LEMMA_REPEAT_MIN
from app.utils import download_button, df_to_csv_bytes, df_to_html_bytes

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(message)s"
)

# --- Reproducibility ---
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

set_seed()

# --- Page Setup ---
st.set_page_config(page_title="French Transition QA Tool", layout="wide")
st.title("French Transition QA Tool (Prototype)")

# --- Configuration Panel ---
with st.expander("Configuration", expanded=False):
    st.write("Tune thresholds. These affect pass/fail logic in real time.")
    max_words = st.number_input(
        "Max transition words", min_value=1, max_value=10, value=MAX_TRANSITION_WORDS,
        help="Maximum words allowed in a transition phrase."
    )
    lemma_min = st.number_input(
        "Lemma repetition min count", min_value=2, max_value=10, value=LEMMA_REPEAT_MIN,
        help="Minimum frequency for a lemma to be considered repetitive."
    )

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file:
    try:
        # Extract text based on file type
        with st.spinner("Extracting text..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                raw_text = extract_text_from_pdf(uploaded_file)
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if not raw_text.strip():
            st.error("No text extracted from the file. Please check the file content.")
        else:
            # Display raw text for debugging
            with st.expander("View Extracted Text", expanded=False):
                st.text_area("Raw Text", raw_text, height=200, disabled=True)

            # Update config dynamically
            import app.config as cfg
            cfg.MAX_TRANSITION_WORDS = max_words
            cfg.LEMMA_REPEAT_MIN = lemma_min

            # --- Evaluation ---
            with st.spinner("Evaluating article..."):
                results_df, summary_stats = evaluate_article(raw_text, article_id=uploaded_file.name)

            # --- Results Table ---
            st.subheader("Evaluation Results")
            if results_df.empty:
                st.info("No transitions detected. Adjust your text/markers or thresholds.")
            else:
                display_cols = [
                    "article_id", "para_idx", "transition_text", 
                    "per-rule pass/fail", "failure_reason", "triggered_rule",
                    "similarity_next", "similarity_prev"
                ]
                st.dataframe(results_df[display_cols], width="stretch")

                # --- Download Options ---
                st.subheader("Downloads")
                csv_b = df_to_csv_bytes(results_df)
                html_b = df_to_html_bytes(results_df)
                download_button(csv_b, f"{uploaded_file.name}_results.csv", "Download CSV")
                download_button(html_b, f"{uploaded_file.name}_results.html", "Download HTML")

                # --- Summary Stats ---
                st.subheader("Summary Stats")
                st.json(summary_stats)

                # --- Failure Types Bar Chart ---
                if "most_common_failure_types" in summary_stats and summary_stats["most_common_failure_types"]:
                    st.markdown("### Breakdown by Triggered Rule")
                    fail_df = pd.DataFrame.from_dict(
                        summary_stats["most_common_failure_types"], orient="index", columns=["Count"]
                    )
                    fail_df = fail_df.sort_values("Count", ascending=False)
                    st.bar_chart(fail_df)
                else:
                    st.write("No failure types to display.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logging.error(f"File processing failed: {str(e)}")
else:
    st.caption("Upload a file to begin")