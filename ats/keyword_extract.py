"""Keyword extraction utilities for ATS alignment."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from openai_utils import DEFAULT_OPENAI_MODEL, get_openai_client

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "will",
    "your",
    "have",
    "work",
    "team",
    "role",
    "skills",
    "required",
    "responsibilities",
    "experience",
}

# Lightweight synonym map intentionally curated for data / product / engineering roles.
SYNONYMS = {
    "sql": {"postgres", "mysql", "redshift"},
    "python": {"pandas", "numpy"},
    "dashboard": {"tableau", "looker", "powerbi"},
    "analysis": {"analytics", "analytical"},
    "manage": {"lead", "led", "oversaw"},
    "project": {"program", "initiative"},
    "communication": {"stakeholder", "presentation"},
    "cloud": {"aws", "gcp", "azure"},
    "automation": {"workflow", "orchestration"},
    "testing": {"qa", "quality"},
}


@dataclass
class KeywordCandidate:
    token: str
    score: float
    synonyms: List[str]


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}")


def normalise(text: str) -> List[str]:
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    return filtered


def _ngram(tokens: Sequence[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _extract_keywords_basic(text: str, max_keywords: int) -> List[KeywordCandidate]:
    tokens = normalise(text)
    if not tokens:
        return []

    unigram_counts = Counter(tokens)
    bigram_tokens = list(_ngram(tokens, 2))
    bigram_counts = Counter(bigram_tokens)
    first_seen: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        first_seen.setdefault(token, idx)
    offset = len(tokens)
    for idx, token in enumerate(bigram_tokens):
        first_seen.setdefault(token, offset + idx)

    candidates: List[KeywordCandidate] = []

    for token, count in unigram_counts.most_common():
        synonyms = sorted(SYNONYMS.get(token, []))
        candidates.append(KeywordCandidate(token=token, score=float(count), synonyms=synonyms))

    for token, count in bigram_counts.most_common():
        if any(part in STOPWORDS for part in token.split()):
            continue
        candidates.append(KeywordCandidate(token=token, score=float(count) + 0.5, synonyms=[]))

    seen: set[str] = set()
    deduped: List[KeywordCandidate] = []
    for candidate in sorted(
        candidates, key=lambda cand: (-cand.score, first_seen.get(cand.token, 0))
    ):
        if candidate.token in seen:
            continue
        seen.add(candidate.token)
        deduped.append(candidate)
        if len(deduped) >= max_keywords:
            break

    return deduped


def extract_keywords_openai(text: str, *, max_keywords: int = 20) -> Optional[List[KeywordCandidate]]:
    """Extract keywords using OpenAI's semantic understanding."""
    client = get_openai_client()
    if client is None:
        return None

    prompt = (
        "Extract the most important keywords and technical skills from this job description for ATS optimization.\n"
        "Focus on technical skills, tools, frameworks, certifications, and action verbs that matter for hiring.\n"
        f"Return up to {max_keywords} keywords as JSON objects like {{\"keyword\": \"python\", \"synonyms\": [], \"importance\": 0.0}}.\n\n"
        "Job description:\n"
        f"{text[:3500]}"
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You extract structured resume keywords for ATS optimisation."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.3,
        )

        result_text = (response.choices[0].message.content or "").strip()
        if not result_text:
            return None

        if "```json" in result_text:
            result_text = result_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in result_text:
            result_text = result_text.split("```", 1)[1].split("```", 1)[0]

        data = json.loads(result_text)
        candidates: List[KeywordCandidate] = []
        for item in data:
            keyword = item.get("keyword", "").lower().strip()
            if not keyword or keyword in STOPWORDS:
                continue
            synonyms = [s.lower().strip() for s in item.get("synonyms", []) if s]
            importance = float(item.get("importance", 0.5))
            candidates.append(
                KeywordCandidate(
                    token=keyword,
                    score=max(importance, 0.1) * 10.0,
                    synonyms=synonyms,
                )
            )

        return candidates[:max_keywords] if candidates else None

    except Exception as exc:
        print(f"OpenAI extraction failed: {exc}. Falling back to basic extraction.")
        return None


def extract_keywords(
    text: str, *, max_keywords: int = 20, use_openai: bool = True
) -> List[KeywordCandidate]:
    """Return ranked keyword candidates."""
    if use_openai:
        ai_candidates = extract_keywords_openai(text, max_keywords=max_keywords)
        if ai_candidates:
            return ai_candidates

    return _extract_keywords_basic(text, max_keywords)
