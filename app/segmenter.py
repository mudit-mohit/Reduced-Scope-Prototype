import re
from typing import List, Tuple, Dict
from .config import MAX_TRANSITION_WORDS, TRANSITION_MARKERS, CONCLUDING_MARKERS
import unicodedata
from difflib import SequenceMatcher

BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
INLINE_CAND_RE = re.compile(r"^([\wÀ-ÖØ-öø-ÿ'\- ]{1,80}?)[,:–—-] ")
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
    raw_text = re.sub(r"\r\n?", "\n", raw_text)
    raw_text = re.sub(r"\t", " ", raw_text)
    blocks = [b.strip() for b in BLOCK_SPLIT_RE.split(raw_text) if b.strip()]
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
    lower = normalize_text(text)
    for m in TRANSITION_MARKERS:
        if lower == normalize_text(m) or lower.startswith(normalize_text(m) + " "):
            return True
    for m in TRANSITION_MARKERS:
        ratio = SequenceMatcher(None, lower, normalize_text(m)).ratio()
        if ratio >= 0.7:
            return True
    wc = len(lower.split())
    if wc <= 8 and (lower.endswith(",") or lower.endswith(":")):
        return True
    return False

def is_concluding_like(text: str) -> bool:
    lower = text.lower().strip()
    return (
        lower in CONCLUDING_MARKERS
        or any(lower.startswith(m + " ") for m in CONCLUDING_MARKERS)
    )

def detect_transitions(blocks: List[str]) -> Tuple[List[TransitionRecord], List[str]]:
    """
    Detect transitions in natural order with correct sequential para_idx.
    """
    transitions: List[TransitionRecord] = []
    new_content_blocks: List[str] = []

    content_counter = 0  # sequential counter for actual content

    for raw_idx, blk in enumerate(blocks):
        blk_lower = blk.lower().strip()

        # Skip metadata blocks
        if blk_lower.startswith("titre:") or blk_lower.startswith("chapeau:") or blk_lower.startswith("article:"):
            continue

        # Increment sequential paragraph counter
        content_counter += 1
        para_idx = content_counter

        # Case 1: standalone transition block
        if is_marker_like(blk):
            transitions.append(
                TransitionRecord(
                    idx_between=para_idx,
                    text=blk.strip(),
                    kind="standalone",
                    is_concluding=is_concluding_like(blk)
                )
            )
            new_content_blocks.append(blk)
            continue

        # Case 2: inline transition inside a block
        m = INLINE_CAND_RE.match(blk)
        if m and is_marker_like(m.group(1)):
            first_sentence = m.group(1).strip()
            transitions.append(
                TransitionRecord(
                    idx_between=para_idx,
                    text=first_sentence,
                    kind="inline",
                    is_concluding=is_concluding_like(first_sentence)
                )
            )
            remainder = blk[len(first_sentence):].strip()
            new_content_blocks.append(first_sentence)
            if remainder:
                new_content_blocks.append(remainder)
            continue

        # Otherwise, keep as content
        new_content_blocks.append(blk)

    return transitions, new_content_blocks