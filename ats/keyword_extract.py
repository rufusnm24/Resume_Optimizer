"""Keyword extraction utilities for ATS alignment."""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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


def extract_keywords_openai(text: str, *, max_keywords: int = 20) -> List[KeywordCandidate]:
    """Extract keywords using OpenAI's semantic understanding."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # Fallback to basic extraction
        return extract_keywords(text, max_keywords=max_keywords)
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Extract the most important keywords and technical skills from this job description for ATS optimization. 
    Focus on:
    - Technical skills and tools
    - Programming languages and frameworks
    - Industry-specific terms
    - Job requirements and qualifications
    - Action-oriented competencies

    Return exactly {max_keywords} keywords as a JSON array of objects with this format:
    [{{"keyword": "python", "synonyms": ["pandas", "numpy"], "importance": 0.95}}]

    Job Description:
    {text[:3000]}"""  # Limit to avoid token limits
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        import json
        result_text = response.choices[0].message.content.strip()
        # Extract JSON from response (handle potential markdown formatting)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        keywords_data = json.loads(result_text)
        
        candidates = []
        for item in keywords_data:
            keyword = item.get("keyword", "").lower().strip()
            if keyword and keyword not in STOPWORDS:
                synonyms = [s.lower().strip() for s in item.get("synonyms", [])]
                importance = float(item.get("importance", 0.5))
                candidates.append(KeywordCandidate(
                    token=keyword,
                    score=importance * 10,  # Scale to match basic extraction scores
                    synonyms=synonyms
                ))
        
        return candidates[:max_keywords]
        
    except Exception as e:
        print(f"OpenAI extraction failed: {e}. Falling back to basic extraction.")
        return extract_keywords(text, max_keywords=max_keywords)


def extract_keywords(text: str, *, max_keywords: int = 20, use_openai: bool = True) -> List[KeywordCandidate]:
    """Return ranked keyword candidates.
    
    Args:
        text: Job description text to extract keywords from
        max_keywords: Maximum number of keywords to return
        use_openai: Whether to use OpenAI for enhanced extraction (falls back to basic if unavailable)
    """
    if use_openai:
        return extract_keywords_openai(text, max_keywords=max_keywords)
    
    # Basic extraction logic (original implementation)
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

    # Promote meaningful bigrams by boosting their score.
    for token, count in bigram_counts.most_common():
        if any(part in STOPWORDS for part in token.split()):
            continue
        synonyms = []
        candidates.append(KeywordCandidate(token=token, score=float(count) + 0.5, synonyms=synonyms))

    # Deduplicate while preserving order (higher score first).
    seen: set[str] = set()
    deduped: List[KeywordCandidate] = []
    for candidate in sorted(candidates, key=lambda cand: (-cand.score, first_seen.get(cand.token, 0))):
        if candidate.token in seen:
            continue
        seen.add(candidate.token)
        deduped.append(candidate)
        if len(deduped) >= max_keywords:
            break

    return deduped
