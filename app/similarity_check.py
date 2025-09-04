from sentence_transformers import SentenceTransformer, util
from .utils import normalize
from .config import TRANSITION_MARKERS
import logging
import re

# Load model once
_model = SentenceTransformer("dangvantuan/sentence-camembert-base")

def _extract_semantic_content(text: str, max_sentences: int = 5, min_chars: int = 20, max_chars: int = 500) -> str:
    """Extract minimal semantic content to avoid overlap."""
    if not text:
        return ""
    sentences = re.split(r"[.!?]+\s+", text.strip())
    content = ". ".join(sentences[:max_sentences]) if sentences else ""
    if len(content) < min_chars and len(sentences) > max_sentences:
        content = ". ".join(sentences[:max_sentences + 1])
    return content[:max_chars]

def _clean_for_similarity(text: str) -> str:
    """Light normalization, preserve French-specific content."""
    if not text:
        return ""
    text = normalize(text).lower().strip()
    text = re.sub(r"[^\w\sàâäéèêëîïôöùûüç]", " ", text)  # stricter punctuation removal
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_known_marker(text: str) -> bool:
    """Check if text is a transition marker, strict matching."""
    t_norm = normalize(text).lower()
    for m in TRANSITION_MARKERS:
        m_norm = normalize(m).lower()
        if t_norm == m_norm or t_norm.startswith(m_norm + " "):
            logging.info(f"[MARKER MATCH] '{t_norm}' matches '{m_norm}'")
            return True
    keywords = ["sportif", "culturel", "conclusion"]
    for kw in keywords:
        if re.search(rf"\b{kw}\b", t_norm):
            logging.info(f"[MARKER KEYWORD] '{kw}' in '{t_norm}'")
            return True
    return False

def compute_similarity(text1: str, text2: str, use_semantic_extraction: bool = True) -> float:
    """Compute cosine similarity between two texts with stable scaling."""
    if not text1 or not text2:
        logging.warning(f"[SIMILARITY WARNING] Empty input: text1='{text1}', text2='{text2}'")
        return 0.0

    # Extract richer context
    if use_semantic_extraction:
        text1_proc = _extract_semantic_content(text1)
        text2_proc = _extract_semantic_content(text2)
    else:
        text1_proc, text2_proc = text1, text2

    # Clean text
    text1_proc = _clean_for_similarity(text1_proc)
    text2_proc = _clean_for_similarity(text2_proc)

    try:
        embeddings = _model.encode(
            [text1_proc, text2_proc],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        similarity_raw = float(util.cos_sim(embeddings[0], embeddings[1]))
        # Rescale [-1,1] → [0,1]
        similarity = (similarity_raw + 1.0) / 2.0

        logging.info(f"[SIMILARITY RESULT] Raw={similarity_raw:.3f}, Scaled={similarity:.3f}")
        return similarity
    except Exception as e:
        logging.error(f"[SIMILARITY ERROR] {str(e)}")
        return 0.0

def compute_transition_similarity(transition: str, prev_paragraph: str, next_paragraph: str) -> tuple[float, float]:
    prev_similarity = compute_similarity(transition, prev_paragraph, use_semantic_extraction=True)
    next_similarity = compute_similarity(transition, next_paragraph, use_semantic_extraction=True)

    # Safety clamp (force everything into [0,1])
    prev_similarity = max(0.0, min(1.0, prev_similarity))
    next_similarity = max(0.0, min(1.0, next_similarity))

    if is_known_marker(transition):
        if "conclusion" in transition.lower():
            pass
        else:
            if next_similarity < 0.25:
                next_similarity = 0.25
            else:
                next_similarity = min(1.0, next_similarity + 0.05)

    logging.info(
        f"[TRANSITION SIMILARITY] '{transition[:30]}...' -> prev={prev_similarity:.3f}, next={next_similarity:.3f}"
    )
    return prev_similarity, next_similarity