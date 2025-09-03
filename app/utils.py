import streamlit as st
import pandas as pd
from io import BytesIO
import re
import spacy
import logging

# Load spaCy model for French
nlp = spacy.load("fr_core_news_sm")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

def df_to_html_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to HTML bytes with basic styling."""
    html = df.to_html(index=False, classes="table table-striped")
    # Add basic CSS for better readability
    styled_html = f"""
    <style>
        .table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .table th {{
            background-color: #f2f2f2;
        }}
        .table-striped tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
    {html}
    """
    return styled_html.encode('utf-8')

def download_button(data: bytes, filename: str, label: str):
    """Create a Streamlit download button for CSV or HTML."""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/csv" if filename.endswith(".csv") else "text/html"
    )

def normalize(text: str) -> str:
    """
    Normalize text by removing stop words and punctuation using spaCy.
    
    Args:
        text: Input text to normalize.
    
    Returns:
        str: Normalized text with stop words and punctuation removed.
    """
    if not text:
        logging.warning("[NORMALIZE WARNING] Empty input text")
        return ""
    
    # Initial regex cleaning for non-standard characters
    text = text.lower().strip()
    text = re.sub(r'[^\w\sàâäéèêëîïôöùûüç]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Process with spaCy to remove stop words and punctuation
    doc = nlp(text)
    normalized = " ".join(token.text for token in doc if not token.is_stop and not token.is_punct)
    
    logging.info(f"[NORMALIZE] Input: '{text[:100]}...' -> Output: '{normalized[:100]}...'")
    return normalized.strip()