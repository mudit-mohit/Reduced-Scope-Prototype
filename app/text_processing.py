from typing import List, Tuple, Dict
from app.segmenter import split_into_blocks, detect_transitions
import logging

def segment_article(raw_text: str, article_id: str) -> Tuple[List[str], List[Dict]]:
    """
    Segment an article into paragraphs and detect transitions using segmenter.py.
    
    Args:
        raw_text: Raw article text.
        article_id: Identifier for the article.
    
    Returns:
        Tuple of (content_blocks, transitions_out).
    """
    blocks = split_into_blocks(raw_text)
    transitions, content_blocks = detect_transitions(blocks)
    transitions_out = []
    for seq, tr in enumerate(transitions, start=1):
        rec = tr.to_dict_base()
        rec["article_id"] = article_id
        rec["transition_seq"] = seq
        transitions_out.append(rec)

    logging.info(f"[SEGMENT ARTICLE] Found {len(transitions)} transitions: {[t['transition_text'] for t in transitions_out]}")
    logging.info(f"[SEGMENT ARTICLE] Transition para_idx: {[t['para_idx'] for t in transitions_out]}")
    return content_blocks, transitions_out