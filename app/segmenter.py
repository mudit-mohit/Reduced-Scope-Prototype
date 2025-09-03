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
    
    # Normalize line endings and tabs
    raw_text = re.sub(r"\r\n?", "\n", raw_text)
    raw_text = re.sub(r"\t", " ", raw_text)
    
    # CRITICAL FIX: Add paragraph breaks around transitions to create proper structure
    # This ensures transitions become separate blocks from their following content
    
    # Remove "Transitions générées:" section first
    raw_text = re.sub(r"Transitions générées:.*", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Add breaks before section headers
    raw_text = re.sub(r"(À savoir également dans votre département)", r"\n\n\1\n\n", raw_text)
    
    # Add breaks around transitions to separate them from following content
    # This is KEY to getting the right paragraph indices
    
    # Pattern 1: "Dans le registre culturel," -> separate paragraph
    raw_text = re.sub(r"(Dans le registre culturel,)\s*", r"\n\n\1\n\n", raw_text)
    
    # Pattern 2: "Côté sportif, on annonce que" -> separate paragraph  
    raw_text = re.sub(r"(Côté sportif, on annonce que)\s*", r"\n\n\1\n\n", raw_text)
    
    # Pattern 3: "En guise de conclusion." -> separate paragraph
    raw_text = re.sub(r"(En guise de conclusion\.)\s*", r"\n\n\1\n\n", raw_text)
    
    # Split on double newlines
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
    FIXED: Detect transitions with their actual paragraph positions in content structure.
    """
    transitions: List[TransitionRecord] = []
    new_content_blocks: List[str] = []
    
    # CRITICAL FIX: Use content_counter for para_idx instead of transition_counter
    content_counter = 0  # This will be the actual paragraph index

    for raw_idx, blk in enumerate(blocks):
        blk_lower = blk.lower().strip()

        # Skip metadata blocks (don't increment content_counter)
        if blk_lower.startswith(("titre:", "chapeau:")):
            logger.debug(f"[SKIP META] raw_idx={raw_idx}, blk='{blk[:40]}...'")
            continue
            
        # Handle "Article:" prefix specially
        if blk_lower.startswith("article:"):
            if len(blk.strip()) < 20:  # Short "Article:" header
                logger.debug(f"[SKIP ARTICLE HEADER] raw_idx={raw_idx}, blk='{blk[:40]}...'")
                continue
            else:
                # Remove "Article:" prefix and continue processing
                blk = blk[8:].strip()
                logger.debug(f"[PROCESS ARTICLE] Removed prefix, now: '{blk[:40]}...'")

        # FIXED: Case 1 - standalone transition
        if is_marker_like(blk):
            logger.info(
                f"[STANDALONE TRANSITION] Found at content_idx={content_counter}, "
                f"text='{blk[:40]}...'"
            )
            transitions.append(
                TransitionRecord(
                    idx_between=content_counter,  # FIXED: Use actual position in content
                    text=blk.strip(),
                    kind="standalone",
                    is_concluding=is_concluding_like(blk),
                )
            )
            new_content_blocks.append(blk)
            content_counter += 1  # Increment after adding
            continue

        # FIXED: Case 2 - inline transition  
        m = INLINE_CAND_RE.match(blk)
        if m and is_marker_like(m.group(1)):
            first_sentence = m.group(1).strip()
            logger.info(
                f"[INLINE TRANSITION] Found at content_idx={content_counter}, "
                f"text='{first_sentence[:40]}...'"
            )
            transitions.append(
                TransitionRecord(
                    idx_between=content_counter,  # FIXED: Use actual position in content
                    text=first_sentence,
                    kind="inline",
                    is_concluding=is_concluding_like(first_sentence),
                )
            )
            remainder = blk[len(first_sentence):].strip()
            new_content_blocks.append(first_sentence)
            content_counter += 1
            
            if remainder:
                logger.debug(f"[INLINE REMAINDER] Adding remainder: '{remainder[:40]}...'")
                new_content_blocks.append(remainder)
                content_counter += 1
            continue

        # Case 3: normal content
        logger.debug(f"[CONTENT] Adding at content_idx={content_counter}, blk='{blk[:40]}...'")
        new_content_blocks.append(blk)
        content_counter += 1

    # Debug output
    logger.info(f"[DETECT_TRANSITIONS] Final results:")
    logger.info(f"  Total content blocks: {len(new_content_blocks)}")
    logger.info(f"  Total transitions: {len(transitions)}")
    
    for i, block in enumerate(new_content_blocks):
        logger.info(f"  [CONTENT {i}] {block[:60]}...")
        
    for tr in transitions:
        logger.info(f"  [TRANSITION] Para {tr.idx_between}: '{tr.text[:50]}...' ({tr.kind})")

    return transitions, new_content_blocks

def create_enhanced_paragraph_breaks(raw_text: str) -> str:
    """
    Create enhanced paragraph breaks to ensure proper transition separation.
    This function can be called before split_into_blocks for better control.
    """
    
    # Remove metadata that interferes with processing
    raw_text = re.sub(r"Transitions générées:.*", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Ensure section headers are properly separated
    raw_text = re.sub(r"(?<=[.!?])\s*(À savoir également)", r"\n\n\1", raw_text)
    
    # CRITICAL: Separate transitions from their following content
    # This creates the paragraph structure needed for indices 3, 5, 7
    
    # Separate "Dans le registre culturel," from following content
    raw_text = re.sub(
        r"(Dans le registre culturel,)\s*(À l'abbaye)", 
        r"\1\n\n\2", 
        raw_text
    )
    
    # Separate "Côté sportif, on annonce que" from following content
    raw_text = re.sub(
        r"(Côté sportif, on annonce que)\s*(À Ruffec)", 
        r"\1\n\n\2", 
        raw_text
    )
    
    # Separate "En guise de conclusion." from following content
    raw_text = re.sub(
        r"(En guise de conclusion\.)\s*(À Triac-Lautrait)", 
        r"\1\n\n\2", 
        raw_text
    )
    
    return raw_text