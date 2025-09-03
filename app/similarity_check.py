from sentence_transformers import SentenceTransformer, util
from .utils import normalize
from .config import TRANSITION_MARKERS
import logging
import re

_model = SentenceTransformer("dangvantuan/sentence-camembert-base")

def _extract_semantic_content(text: str, max_sentences: int = 12, min_chars: int = 60, max_chars: int = 1000) -> str:
    """Extract meaningful semantic content with guaranteed minimum length."""
    if not text:
        return ""
    sentences = re.split(r"[.!?]+\s+", text.strip())
    content = ". ".join(sentences[:max_sentences])
    if len(content) < min_chars and len(sentences) > 2:
        content = ". ".join(sentences[:3])
    return content[:max_chars]

def _clean_for_similarity(text: str) -> str:
    """Clean text for similarity computation."""
    if not text:
        return ""
    text = normalize(text).lower().strip()
    text = re.sub(r'[^\w\sàâäéèêëîïôöùûüç\-\',]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_known_marker(text: str) -> bool:
    """Check if text matches a known transition marker."""
    t_norm = normalize(text)
    for m in TRANSITION_MARKERS:
        m_norm = normalize(m)
        if t_norm == m_norm or t_norm.startswith(m_norm) or m_norm in t_norm:
            return True
    return any(kw in t_norm for kw in ["sportif", "culturel", "conclusion"])

def compute_similarity(text1: str, text2: str, use_semantic_extraction: bool = True) -> float:
    """Compute cosine similarity between two texts."""
    if not text1 or not text2:
        logging.warning(f"[SIMILARITY WARNING] Empty input: text1='{text1}', text2='{text2}'")
        return 0.0

    text1_clean = _extract_semantic_content(text1) if use_semantic_extraction else text1
    text2_clean = _extract_semantic_content(text2) if use_semantic_extraction else text2

    if len(text1_clean.split()) < 3:
        text1_clean = text1
    if len(text2_clean.split()) < 3:
        text2_clean = text2

    text1_clean = _clean_for_similarity(text1_clean)
    text2_clean = _clean_for_similarity(text2_clean)

    if text1_clean == text2_clean:
        logging.warning(f"[SIMILARITY WARNING] Identical texts: '{text1_clean}'")
        return 1.0

    logging.info(f"[SIMILARITY INPUT] Comparing: text1='{text1_clean[:80]}...', text2='{text2_clean[:80]}...'")
    try:
        embeddings = _model.encode(
            [text1_clean, text2_clean],
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        # Apply boost for known markers
        if is_known_marker(text1):
            similarity += 0.15  # Match your boost
        similarity_score = max(0.0, min(1.0, similarity))
        logging.info(f"[SIMILARITY RESULT] Raw: {similarity:.3f}, Final: {similarity_score:.3f}")
        return similarity_score
    except Exception as e:
        logging.error(f"[SIMILARITY ERROR] Failed: {str(e)}")
        return 0.0

def compute_transition_similarity(transition: str, prev_paragraph: str, next_paragraph: str) -> tuple[float, float]:
    """Compute transition similarities with proper context."""
    prev_similarity = compute_similarity(transition, prev_paragraph, use_semantic_extraction=True)
    next_similarity = compute_similarity(transition, next_paragraph, use_semantic_extraction=True)
    # Boost next similarity for markers
    if is_known_marker(transition):
        next_similarity = min(1.0, next_similarity + 0.05)
    logging.info(f"[TRANSITION SIMILARITY] '{transition[:30]}...' -> prev={prev_similarity:.3f}, next={next_similarity:.3f}")
    return prev_similarity, next_similarity