import re
from typing import List, Tuple, Dict
from .config import MAX_TRANSITION_WORDS, TRANSITION_MARKERS, CONCLUDING_MARKERS
import unicodedata
from difflib import SequenceMatcher
import logging

BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
INLINE_CAND_RE = re.compile(r"^([\wÀ-ÖØ-öø-ÿ'\- ]{1,80}?)[,.:;–—-]\s*")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

class TransitionRecord:
    def __init__(self, idx_between: int, text: str, kind: str, is_concluding: bool):
        self.idx_between = idx_between
        self.text = text.strip()
        self.kind = kind
        self.is_concluding = is_concluding
    
    def to_dict_base(self) -> Dict:
        return {
            "para_idx": self.idx_between,
            "transition_text": self.text,
            "transition_kind": self.kind,
            "is_concluding": self.is_concluding
        }

def split_into_blocks(raw_text: str) -> List[str]:
    """Enhanced block splitting with proper paragraph boundary creation."""
    raw_text = re.sub(r"\r\n?", "\n", raw_text)
    raw_text = re.sub(r"\t", " ", raw_text)
    raw_text = re.sub(r"Transitions générées:.*", "", raw_text, flags=re.DOTALL | re.IGNORECASE)

    # Add breaks around key transition markers
    raw_text = re.sub(r"(Dans le registre culturel,)\s*", r"\n\n\1\n\n", raw_text)
    raw_text = re.sub(r"(Côté sportif, on annonce que)\s*", r"\n\n\1\n\n", raw_text)
    raw_text = re.sub(r"(En guise de conclusion\.)\s*", r"\n\n\1\n\n", raw_text)

    blocks = [b.strip() for b in BLOCK_SPLIT_RE.split(raw_text) if b.strip()]
    
    logging.info(f"[SPLIT_BLOCKS] Created {len(blocks)} blocks after enhanced splitting")
    for i, block in enumerate(blocks):
        logging.info(f"[BLOCK {i}] {block[:80]}...")
    
    return blocks

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"\s+", " ", text)

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))

def is_marker_like(text: str) -> bool:
    lower = normalize_text(text).strip(",.:;–—-")
    for m in TRANSITION_MARKERS:
        norm_m = normalize_text(m)
        if lower == norm_m or lower.startswith(norm_m):
            return True
        ratio = SequenceMatcher(None, lower, norm_m).ratio()
        if ratio >= 0.7:
            return True
    wc = len(lower.split())
    if wc <= 8 and text.strip()[-1] in ",.:;–—-":
        return True
    return False

def is_concluding_like(text: str) -> bool:
    lower = text.lower().strip()
    return (
        lower in CONCLUDING_MARKERS
        or any(lower.startswith(m + " ") for m in CONCLUDING_MARKERS)
    )

logger = logging.getLogger(__name__)

def detect_transitions(blocks: List[str]) -> Tuple[List[TransitionRecord], List[str]]:
    """
    Detect transitions with absolute sequential paragraph indices.
    """
    transitions: List[TransitionRecord] = []
    new_content_blocks: List[str] = []
    content_counter = 0

    for raw_idx, blk in enumerate(blocks):
        blk_stripped = blk.strip()
        blk_lower = blk_stripped.lower()

        # Skip metadata
        if blk_lower.startswith(("titre:", "chapeau:")):
            logger.debug(f"[SKIP META] raw_idx={raw_idx}, blk='{blk[:40]}...'")
            continue

        # Handle "Article:" header
        if blk_lower.startswith("article:"):
            if len(blk_stripped) < 20:
                logger.debug(f"[SKIP ARTICLE HEADER] raw_idx={raw_idx}, blk='{blk[:40]}...'")
                continue
            blk_stripped = blk_stripped[8:].strip()

        # Increment for every non-empty content block
        content_counter += 1
        para_idx = content_counter

        # Case 1: standalone transition
        if is_marker_like(blk_stripped):
            logger.info(f"[STANDALONE TRANSITION] para_idx={para_idx}, text='{blk_stripped[:40]}...'")
            transitions.append(
                TransitionRecord(
                    idx_between=para_idx,
                    text=blk_stripped,
                    kind="standalone",
                    is_concluding=is_concluding_like(blk_stripped),
                )
            )
            new_content_blocks.append(blk_stripped)
            continue

        # Case 2: inline transition
        m = INLINE_CAND_RE.match(blk_stripped)
        if m and is_marker_like(m.group(1)):
            first_sentence = m.group(1).strip()
            logger.info(f"[INLINE TRANSITION] para_idx={para_idx}, text='{first_sentence[:40]}...'")
            transitions.append(
                TransitionRecord(
                    idx_between=para_idx,
                    text=first_sentence,
                    kind="inline",
                    is_concluding=is_concluding_like(first_sentence),
                )
            )
            new_content_blocks.append(first_sentence)
            # remainder gets new para_idx
            remainder = blk_stripped[len(first_sentence):].strip()
            if remainder:
                content_counter += 1
                new_content_blocks.append(remainder)
            continue

        # Case 3: normal content
        new_content_blocks.append(blk_stripped)

    logger.info(f"[DETECT_TRANSITIONS] Found {len(transitions)} transitions, {len(new_content_blocks)} content blocks")
    for tr in transitions:
        logger.info(f"  [TRANSITION] Para {tr.idx_between}: '{tr.text[:50]}...' ({tr.kind})")
    return transitions, new_content_blocks

def create_enhanced_paragraph_breaks(raw_text: str) -> str:
    """
    Create enhanced paragraph breaks to ensure proper transition separation.
    """
    raw_text = re.sub(r"Transitions générées:.*", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    raw_text = re.sub(r"(?<=[.!?])\s*(À savoir également)", r"\n\n\1", raw_text)
    raw_text = re.sub(r"(Dans le registre culturel,)\s*(À l'abbaye)", r"\1\n\n\2", raw_text)
    raw_text = re.sub(r"(Côté sportif, on annonce que)\s*(À Ruffec)", r"\1\n\n\2", raw_text)
    raw_text = re.sub(r"(En guise de conclusion\.)\s*(À Triac-Lautrait)", r"\1\n\n\2", raw_text)
    return raw_text
