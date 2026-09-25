"""The shared lexical rule for every memory search surface.

qichi.memory.retriever already matches three-character windows when it looks
for memory records.  The verbatim detail timeline has to answer the same
question -- does this text mention what the user just said? -- with the same
rule, otherwise a topic that is retrievable as a memory record stays invisible
in the details that actually hold the original words.

Two guards keep the rule from over-reaching: a query term shorter than the
window is matched whole, and a candidate needs more than one window hit before
a fragment counts as identified, so a single common three-character window
cannot drag an unrelated episode into context.
"""

from __future__ import annotations

from typing import Sequence

MAX_QUERY_FRAGMENTS = 64
MIN_MATCH_FRAGMENTS = 2
WINDOW = 3


def query_fragments(terms: Sequence[str], *, limit: int = MAX_QUERY_FRAGMENTS) -> tuple[str, ...]:
    """Turn free text into the bounded windows used for lexical matching."""

    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of strings")
    fragments: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not isinstance(term, str):
            raise TypeError("terms must be a sequence of strings")
        text = term.strip()
        if not text:
            continue
        if len(text) < WINDOW:
            windows = (text,)
        else:
            windows = tuple(text[index : index + WINDOW] for index in _window_starts(len(text), limit))
        for window in windows:
            if window not in seen:
                seen.add(window)
                fragments.append(window)
                if len(fragments) >= limit:
                    return tuple(fragments)
    return tuple(fragments)


def _window_starts(length: int, limit: int) -> tuple[int, ...]:
    """Every window start when they fit, otherwise an even sample across the whole text.

    2026-09-17: the cap used to take the *first* limit windows, so a long message's tail never
    took part in matching at all.  Real case: a 76-character turn put
    "什么时候想要我" at character 70, the sample stopped at 66, the fragment holding that
    line scored 1 and was filtered out by the >= 2 threshold.  The cap bounds work; it must
    not let the first half of a sentence decide what the sentence is about.
    """

    total = length - WINDOW + 1
    if total <= limit:
        return tuple(range(total))
    step = total / limit
    return tuple(sorted({min(total - 1, int(index * step)) for index in range(limit)}))


def longest_shared_run(text: str, candidate: str, *, cap: int = 32) -> int:
    """Length of the longest run of characters the two strings share.

    2026-09-17: 排序需要「精确」信号。窗口命中数只说明「有几个三字词重合」，
    通用词（兔子/今天/凌晨）会让长片段占便宜；最长连续串能认出「他说的就是这句」——
    真机里「什么时候想要我」与存下的「什么时候会想要我呀」最长共同串是 4 个字，
    而无关片段的共同串都短于它。**只用于排序**，不做任何授权判断。
    """

    if not isinstance(text, str) or not isinstance(candidate, str):
        raise TypeError("text and candidate must be strings")
    if not text or not candidate:
        return 0
    shorter, longer = (text, candidate) if len(text) <= len(candidate) else (candidate, text)
    best = 0
    for start in range(len(shorter)):
        limit = min(cap, len(shorter) - start, len(longer))
        if limit <= best:
            break
        for size in range(best + 1, limit + 1):
            if shorter[start : start + size] not in longer:
                break
            best = size
    return best


def match_count(text: str, fragments: Sequence[str]) -> int:
    """Count how many distinct windows of the query occur in the candidate."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sum(1 for fragment in fragments if fragment in text)


def matches(text: str, fragments: Sequence[str]) -> bool:
    """Decide whether a candidate mentions the query at all."""

    if not fragments:
        return False
    return match_count(text, fragments) >= min(MIN_MATCH_FRAGMENTS, len(fragments))
