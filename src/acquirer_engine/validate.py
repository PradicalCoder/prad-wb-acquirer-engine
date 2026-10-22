"""Code checks on each one-page rationale. No LLM involved.

validate_page() returns a list of errors, each written as an instruction the
analyst can act on ("16.2 in valuation_note is not in the evidence pack ...").
An empty list means the page passes. similar_theses() compares the 10 pages
with each other, so no two read like copies.

Why code and not an LLM critic: everything checkable is checked exactly here
(numbers, deal IDs, risks, conviction, wording), and code never "finds
something" just because it was asked to look.
"""

import math
import re
from collections import Counter
from itertools import combinations

from acquirer_engine.config import Validation
from acquirer_engine.evidence import EvidencePack
from acquirer_engine.schemas import AnalystNotes, OnePagerDraft

DEAL_ID = re.compile(r"MA-\d{4}-\d{4}")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CONVICTION_STEP = {"Low": 0, "Medium": 1, "High": 2}


def validate_page(
    page: OnePagerDraft, pack: EvidencePack, notes: AnalystNotes, cfg: Validation
) -> list[str]:
    """All problems with one page; an empty list means it passes."""
    errors = []
    prose = _prose_fields(page)

    if page.acquirer != pack.buyer:
        errors.append(f"acquirer must be '{pack.buyer}', not '{page.acquirer}'.")

    # Numbers and deal IDs must come from the evidence pack.
    allowed_numbers = _pack_numbers(pack)
    allowed_ids = set(DEAL_ID.findall(pack.model_dump_json()))
    for field, text in prose.items():
        for written in NUMBER.findall(DEAL_ID.sub(" ", text)):
            if not _grounded(written, allowed_numbers):
                errors.append(f"{written} in {field} is not in the evidence pack. "
                              f"Use only numbers from the pack, copied (rounding is fine).")
        for deal_id in DEAL_ID.findall(text):
            if deal_id not in allowed_ids:
                errors.append(f"{deal_id} in {field} is not in the evidence pack.")

    # Risks must be the pack's risks, with the pack's lens and status.
    pack_risks = {risk.name: risk for risk in pack.risks}
    for risk in page.risks:
        known = pack_risks.get(risk.name)
        if known is None:
            errors.append(f"risk '{risk.name}' is not one of this buyer's risks: "
                          f"{sorted(pack_risks)}.")
        elif (risk.lens, risk.status) != (known.lens, known.status):
            errors.append(f"risk '{risk.name}' must be lens {known.lens}, status {known.status}.")
        elif not set(risk.evidence_ids) <= set(known.evidence_ids):
            errors.append(f"risk '{risk.name}' cites deals that are not its evidence: "
                          f"use only {known.evidence_ids}.")

    # Conviction: at most one level from the rule, and only with a stated reason.
    rule, written = pack.conviction.level, page.conviction.level
    step = CONVICTION_STEP[written] - CONVICTION_STEP[rule]
    planned = {"none": 0, "up": 1, "down": -1}[notes.conviction_adjustment.direction]
    if abs(step) > 1:
        errors.append(f"conviction may move at most one level from the rule ({rule}).")
    elif step != planned:
        errors.append(f"conviction is {written} but the rule says {rule} and your notes "
                      f"planned '{notes.conviction_adjustment.direction}'. Use {rule}, or "
                      f"state the change and its reason in the notes.")

    # Wording that could describe any buyer.
    all_text = " ".join(prose.values()).lower()
    for phrase in cfg.banned_phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", all_text):
            errors.append(f"remove the generic phrase '{phrase}'; say something specific "
                          f"to this buyer instead.")

    words = len(all_text.split())
    if not cfg.page_words_min <= words <= cfg.page_words_max:
        errors.append(f"the page has {words} words of prose; write "
                      f"{cfg.page_words_min}-{cfg.page_words_max}.")
    return errors


def similar_theses(pages: dict[str, OnePagerDraft], cfg: Validation) -> list[tuple[str, str, float]]:
    """Pairs of buyers whose fit theses read alike: (buyer, buyer, similarity).

    Similarity is the cosine of TF-IDF word vectors: words shared by two theses
    count more when they are rare across all ten. 0 = no shared words, 1 = same text.
    """
    words = {buyer: Counter(re.findall(r"[a-z]{3,}", page.fit_thesis.lower()))
             for buyer, page in pages.items()}
    docs_with = Counter(word for counts in words.values() for word in counts)
    idf = {word: math.log(len(pages) / n) + 1 for word, n in docs_with.items()}
    vectors = {buyer: {w: c * idf[w] for w, c in counts.items()} for buyer, counts in words.items()}

    pairs = []
    for a, b in combinations(sorted(vectors), 2):
        score = _cosine(vectors[a], vectors[b])
        if score > cfg.max_thesis_similarity:
            pairs.append((a, b, round(score, 2)))
    return pairs


# --- Helpers ----------------------------------------------------------------------

def _prose_fields(page: OnePagerDraft) -> dict[str, str]:
    """Every piece of prose on the page, by field name. Public-knowledge claims are
    excluded: they are labeled as not from the data."""
    fields = {
        "overview": page.overview,
        "fit_thesis": page.fit_thesis,
        "precedent_note": page.precedent_note,
        "valuation_note": page.valuation_note,
        "conviction.rationale": page.conviction.rationale,
    }
    for i, risk in enumerate(page.risks):
        fields[f"risks[{i}].text"] = risk.text
    return fields


def _pack_numbers(pack: EvidencePack) -> set[float]:
    """Every number in the pack, plus each one x100 (shares are stored as 0.12 but
    written as 12%). Numbers inside the pack's text (e.g. "$349.2M") count too."""
    found: set[float] = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.update({float(value), float(value) * 100})
        elif isinstance(value, str):
            for written in NUMBER.findall(DEAL_ID.sub(" ", value)):
                found.add(float(written.replace(",", "")))
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(pack.model_dump())
    return found


def _grounded(written: str, allowed: set[float]) -> bool:
    """True if `written` equals some pack number rounded to the decimals written.
    Example: "2.5" matches 2.49 (one decimal); "84" matches 84.39; "18" matches 0.18 x 100."""
    value = float(written.replace(",", ""))
    decimals = len(written.split(".")[1]) if "." in written else 0
    return any(round(number, decimals) == value for number in allowed)


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(weight * b.get(word, 0.0) for word, weight in a.items())
    norm = math.sqrt(sum(w * w for w in a.values())) * math.sqrt(sum(w * w for w in b.values()))
    return dot / norm if norm else 0.0
