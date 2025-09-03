from sentence_transformers import SentenceTransformer, util
from .utils import normalize
from .config import TRANSITION_MARKERS
import logging
import re

# Load model once
_model = SentenceTransformer("dangvantuan/sentence-camembert-base")


def _extract_semantic_content(text: str, max_sentences: int = 5, max_chars: int = 500) -> str:
    """Extract meaningful semantic content without truncating too aggressively."""
    if not text:
        return ""
    sentences = re.split(r"[.!?]+\s+", text.strip())
    content = ". ".join(sentences[:max_sentences])
    return content[:max_chars]


def _clean_for_similarity(text: str) -> str:
    """Light normalization, keep accents/punctuation for French model."""
    if not text:
        return ""
    text = normalize(text).lower().strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_known_marker(text: str) -> bool:
    """Check if text matches a known transition marker."""
    t_norm = normalize(text)
    return any(normalize(m) == t_norm for m in TRANSITION_MARKERS)


def compute_similarity(text1: str, text2: str, use_semantic_extraction: bool = True) -> float:
    """Compute cosine similarity between two texts with fallback if extraction is too short."""
    if not text1 or not text2:
        logging.warning(f"[SIMILARITY WARNING] Empty input: text1='{text1}', text2='{text2}'")
        return 0.0

    # Extract and clean
    text1_clean = _extract_semantic_content(text1) if use_semantic_extraction else text1
    text2_clean = _extract_semantic_content(text2) if use_semantic_extraction else text2

    # Fallback: if extracted text too short, use original
    if len(text1_clean.split()) < 3 or len(text1_clean) < 20:
        text1_clean = text1
    if len(text2_clean.split()) < 3 or len(text2_clean) < 20:
        text2_clean = text2

    text1_clean = _clean_for_similarity(text1_clean)
    text2_clean = _clean_for_similarity(text2_clean)

    # Skip similarity check for identical strings
    if text1_clean == text2_clean:
        logging.warning(f"[SIMILARITY WARNING] Identical texts: '{text1_clean}'")
        return 0.0

    try:
        embeddings = _model.encode(
            [text1_clean, text2_clean],
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        # For known markers, enforce a stronger minimum
        if is_known_marker(text1):
            similarity = max(similarity, 0.45)

        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logging.error(f"[SIMILARITY ERROR] Failed: {str(e)}")
        return 0.0


def compute_transition_similarity(transition: str, prev_paragraph: str, next_paragraph: str) -> tuple[float, float]:
    """Compute transition similarities with proper context."""
    prev_similarity = compute_similarity(transition, prev_paragraph, use_semantic_extraction=True)
    next_similarity = compute_similarity(transition, next_paragraph, use_semantic_extraction=True)

    # Boost next similarity slightly for markers (bias forward)
    if is_known_marker(transition):
        next_similarity = min(1.0, next_similarity + 0.05)

    logging.info(
        f"[TRANSITION SIMILARITY] '{transition[:30]}...' -> prev={prev_similarity:.3f}, next={next_similarity:.3f}"
    )
    return prev_similarity, next_similarity