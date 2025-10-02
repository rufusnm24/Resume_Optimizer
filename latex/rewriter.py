"""LaTeX resume optimizer that respects formatting constraints."""
from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ats.keyword_extract import SYNONYMS, KeywordCandidate
from openai_utils import DEFAULT_OPENAI_MODEL, get_openai_client
from .ast_parser import parse_document


@dataclass
class RewriteResult:
    optimized_tex: str
    keyword_map: Dict[str, Dict[str, int]]
    diff: str


def _replace_synonym(text: str, synonym: str, keyword: str) -> Tuple[str, bool]:
    lowered = text.lower()
    idx = lowered.find(synonym)
    if idx == -1:
        return text, False
    # Ensure we are not touching LaTeX macros.
    if "\\" in text[max(0, idx - 2) : idx + len(synonym) + 2]:
        return text, False
    new_text = text[:idx] + keyword + text[idx + len(synonym) :]
    return new_text, True


def _optimize_bullet_openai(
    bullet_content: str,
    keywords: Sequence[KeywordCandidate],
    *,
    strict: bool = False,
) -> str:
    """Use OpenAI to rewrite a bullet point to include keywords naturally."""
    client = get_openai_client()
    if client is None:
        return bullet_content

    keyword_list = [kw.token for kw in keywords[:10]]  # Limit to top 10 keywords

    prompt = f"""Rewrite this resume bullet point to naturally incorporate relevant keywords while maintaining professional quality and truthfulness.

IMPORTANT CONSTRAINTS:
- Keep the same core achievements and responsibilities
- Maintain professional, action-oriented language
- {'Length must stay within +/- 10 characters of original' if strict else 'Length should stay between 50 and 200 characters'}
- Use strong action verbs
- Include specific metrics/numbers if present in original
- Do NOT fabricate achievements or add false information

Target keywords (use 1-3 if relevant): {', '.join(keyword_list)}

Original: {bullet_content}

Rewritten:"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite resume bullets to improve ATS alignment while staying factual.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.4,
        )

        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            return bullet_content

        # Validate length constraint for strict mode
        if strict and abs(len(rewritten) - len(bullet_content)) > 10:
            return bullet_content

        return rewritten

    except Exception as exc:
        print(f"OpenAI bullet rewrite failed: {exc}")
        return bullet_content


def optimize_resume(
    tex_content: str,
    keywords: Sequence[KeywordCandidate],
    *,
    strict: bool = False,
    use_openai: bool = True,
) -> RewriteResult:
    document = parse_document(tex_content)
    usage = Counter()

    before_counts: Dict[str, int] = {
        candidate.token: tex_content.lower().count(candidate.token) for candidate in keywords
    }

    for idx, bullet in enumerate(document.bullets):
        original = bullet.content
        updated = original

        # Try OpenAI optimization first if enabled
        if use_openai:
            ai_optimized = _optimize_bullet_openai(original, keywords, strict=strict)
            if ai_optimized != original:
                updated = ai_optimized
                # Update usage counts based on OpenAI rewrite
                for candidate in keywords:
                    count = updated.lower().count(candidate.token)
                    usage[candidate.token] += count

                if updated != original:
                    document.replace_bullet(idx, updated)
                continue

        # Fallback to original logic if OpenAI not used or failed
        per_bullet = Counter(
            {candidate.token: updated.lower().count(candidate.token) for candidate in keywords}
        )
        for candidate in keywords:
            if usage[candidate.token] >= 2:
                continue
            if per_bullet[candidate.token] >= 1:
                usage[candidate.token] += per_bullet[candidate.token]
                continue
            synonyms = SYNONYMS.get(candidate.token, set()) | set(candidate.synonyms)
            replaced = False
            for synonym in synonyms:
                new_text, success = _replace_synonym(updated, synonym, candidate.token)
                if success:
                    if strict and abs(len(new_text) - len(original)) > 10:
                        continue
                    updated = new_text
                    usage[candidate.token] += 1
                    per_bullet[candidate.token] += 1
                    replaced = True
                    break
            if replaced:
                continue
            if not strict and usage[candidate.token] < 2:
                addition = f" ({candidate.token})"
                new_text = updated + addition
                updated = new_text
                usage[candidate.token] += 1
                per_bullet[candidate.token] += 1
        if strict and abs(len(updated) - len(original)) > 10:
            updated = original
        if updated != original:
            document.replace_bullet(idx, updated)

    optimized_tex = document.render()
    optimized_counts: Dict[str, int] = {
        candidate.token: optimized_tex.lower().count(candidate.token) for candidate in keywords
    }
    keyword_map = {
        candidate.token: {"before": before_counts[candidate.token], "after": optimized_counts[candidate.token]}
        for candidate in keywords
    }
    diff = "\n".join(
        difflib.unified_diff(
            tex_content.splitlines(),
            optimized_tex.splitlines(),
            fromfile="original.tex",
            tofile="main_optimized.tex",
            lineterm="",
        )
    )
    return RewriteResult(optimized_tex=optimized_tex, keyword_map=keyword_map, diff=diff)
