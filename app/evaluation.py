import pandas as pd
import spacy
import re
from app.text_processing import segment_article
from app.similarity_check import compute_transition_similarity, is_known_marker
from app.config import MAX_TRANSITION_WORDS, CONCLUDING_MARKERS, STOP_LEMMAS
from app.utils import normalize
import logging
from collections import Counter

nlp = spacy.load("fr_core_news_md")


def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing spaces and special characters."""
    text = text.replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def lemmatize_text(text: str, nlp_model=nlp) -> list[str]:
    """Lemmatize text and return non-stopword lemmas."""
    doc = nlp_model(preprocess_text(text))
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    logging.info(f"[LEMMATIZE] Text: '{text[:50]}...' -> Lemmas: {lemmas}")
    return lemmas

def evaluate_thematic_cohesion(
    transition: str,
    prev_similarity: float,
    next_similarity: float,
    min_threshold: float = 0.25,
    min_difference: float = 0.05,
    tolerance: float = 1e-3
) -> tuple[bool, str | None]:
    """Evaluate if a transition demonstrates good thematic cohesion."""
    if next_similarity < min_threshold:
        reason = f"Next similarity too low ({next_similarity:.2f} < {min_threshold})"
        logging.info(f"[COHESION FAIL] {reason}")
        return False, reason

    difference = next_similarity - prev_similarity
    if difference + tolerance < min_difference:
        reason = (
            f"Insufficient forward bias: next={next_similarity:.2f}, "
            f"prev={prev_similarity:.2f}, diff={difference:.2f}"
        )
        logging.info(f"[COHESION FAIL] {reason}")
        return False, reason

    if any(normalize(marker) in normalize(transition) for marker in ["finir", "terminer"]):
        if next_similarity + tolerance < 0.2:
            reason = f"Concluding transition needs stronger forward connection: {next_similarity:.2f}"
            logging.info(f"[COHESION FAIL] {reason}")
            return False, reason

    logging.info(
        f"[COHESION PASS] Good thematic cohesion: "
        f"next={next_similarity:.3f}, prev={prev_similarity:.3f}, diff={difference:.3f}"
    )
    return True, None

def evaluate_article(raw_text: str, article_id: str) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate transitions in the article based on rules.
    """
    paragraphs, transitions = segment_article(raw_text, article_id)

    if not transitions:
        logging.warning(f"[EVALUATE] No transitions detected for article '{article_id}'.")
        empty_df = pd.DataFrame(
            columns=[
                "article_id",
                "para_idx",
                "transition_text",
                "per-rule pass/fail",
                "failure_reason",
                "triggered_rule",
                "similarity_next",
                "similarity_prev",
            ]
        )
        summary_stats = {
            "compliance_pct": None,
            "failure_types": {},
            "top_lemmas": [],
        }
        return empty_df, summary_stats

    results = []
    full_text = " ".join(paragraphs)
    lemma_counts = Counter(lemmatize_text(full_text))
    logging.info(f"[LEMMA COUNTS] {dict(lemma_counts)}")

    for i, transition in enumerate(transitions):
        transition_text = transition.get("transition_text", "").strip()
        para_idx = transition.get("para_idx", i + 1)
        article_id = transition.get("article_id", article_id)

        logging.info(f"=== PROCESSING TRANSITION {i} ===")
        logging.info(f"Transition: '{transition_text}'")
        logging.info(f"Para index: {para_idx}")

        pass_fail = "Pass"
        failure_reasons = []
        triggered_rules = []
        similarity_prev = None
        similarity_next = None

        # Rule 1: Word limit check
        words = [
            w.strip()
            for w in re.split(r"\s+", transition_text)
            if w.strip() and not re.match(r"[^\w\sàâäéèêëîïôöùûüç]", w)
        ]
        word_count = len(words)
        logging.info(f"Word count: {word_count} (words: {words})")
        if word_count > MAX_TRANSITION_WORDS:
            pass_fail = "Fail"
            failure_reasons.append(
                f"Transition exceeds {MAX_TRANSITION_WORDS} words ({word_count})"
            )
            triggered_rules.append("WORD_LIMIT")

        # Rule 2: Lemma repetition check
        if not is_known_marker(transition_text):
            transition_lemmas = lemmatize_text(transition_text)
            repeated_lemmas = [
                lemma
                for lemma in set(transition_lemmas)
                if lemma not in STOP_LEMMAS and lemma_counts.get(lemma, 0) >= 3
            ]
            if len(repeated_lemmas) >= 2:
                pass_fail = "Fail"
                failure_reasons.append(
                    f"Too many repeated lemmas: {', '.join(repeated_lemmas)}"
                )
                triggered_rules.append("REPEATED_LEMMAS")

        # Rule 3: Thematic cohesion
        prev_para = paragraphs[para_idx - 2] if para_idx - 2 >= 0 else None
        next_para = paragraphs[para_idx] if para_idx < len(paragraphs) else None

        if prev_para and next_para:
            similarity_prev, similarity_next = compute_transition_similarity(
                transition_text, prev_para, next_para
            )

            if not evaluate_thematic_cohesion(
                transition_text, similarity_prev or 0, similarity_next or 0
            ):
                pass_fail = "Fail"
                if similarity_next < 0.25:
                    failure_reasons.append(
                        f"Next similarity too low ({similarity_next:.2f} < 0.25)"
                    )
                else:
                    diff = (similarity_next or 0) - (similarity_prev or 0)
                    failure_reasons.append(
                        f"Insufficient forward bias (diff={diff:.2f}, next={similarity_next:.2f}, prev={similarity_prev:.2f})"
                    )
                triggered_rules.append("LOW_SIMILARITY")

            if similarity_prev > 0.9:
                logging.warning(
                    f"[EVALUATE WARNING] High prev_similarity={similarity_prev:.3f}"
                )

        # Rule 4: Concluding transition check
        is_concluding = any(
            normalize(marker) in normalize(transition_text)
            for marker in CONCLUDING_MARKERS
        )
        if is_concluding and i != len(transitions) - 1:
            pass_fail = "Fail"
            failure_reasons.append("Concluding transition not at end")
            triggered_rules.append("CONCLUDING_NOT_LAST")
        elif is_concluding:
            triggered_rules.append("CONCLUDING")

        results.append(
            {
                "article_id": article_id,
                "para_idx": para_idx,
                "transition_text": transition_text,
                "per-rule pass/fail": pass_fail,
                "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
                "triggered_rule": "; ".join(triggered_rules) if triggered_rules else None,
                "similarity_next": round(similarity_next, 3) if similarity_next is not None else 0.0,
                "similarity_prev": round(similarity_prev, 3) if similarity_prev is not None else 0.0,
            }
        )

    results_df = pd.DataFrame(results)
    compliance_pct = (
        (results_df["per-rule pass/fail"] == "Pass").mean() * 100
        if len(results_df) > 0
        else None
    )
    failure_types = (
        results_df[results_df["per-rule pass/fail"] == "Fail"]["triggered_rule"]
        .value_counts()
        .to_dict()
    )
    top_lemmas = [
        (lemma, count)
        for lemma, count in lemma_counts.most_common()
        if lemma not in STOP_LEMMAS and count >= 3
    ]

    summary_stats = {
        "compliance_pct": round(compliance_pct, 2)
        if compliance_pct is not None
        else None,
        "failure_types": failure_types,
        "top_lemmas": top_lemmas,
    }

    logging.info(f"Evaluation results: {results_df.to_dict()}")
    logging.info(f"Summary stats: {summary_stats}")
    return results_df, summary_stats
