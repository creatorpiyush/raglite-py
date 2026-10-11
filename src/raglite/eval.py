"""Retrieval evaluation: recall@k, MRR and hit rate per retrieval mode.

Each question lists what a relevant result looks like: a string the passage
contains (case and whitespace are ignored), or a location such as
``{"source": "policy.pdf", "page": 3}`` or ``{"source": "notes.md", "chunk": 2}``.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .errors import ConfigError
from .types import SearchResult

Expected = Union[str, Dict[str, Any]]


@dataclass
class EvalQuestion:
    question: str
    expected: List[Expected]


@dataclass
class EvalRow:
    mode: str  # the retrieval mode, with "+rerank" when reranked
    recall: float  # mean share of each question's expected items found in the top k
    mrr: float  # mean reciprocal rank of the first relevant result (0 when none)
    hitRate: float  # share of questions with at least one relevant result in the top k
    missed: List[str] = field(default_factory=list)  # questions with no relevant result


@dataclass
class EvalReport:
    topK: int
    questions: int
    results: List[EvalRow]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def matches_expected(hit: SearchResult, expected: Expected) -> bool:
    """True when ``hit`` is what ``expected`` describes."""
    if isinstance(expected, str):
        return _normalize(expected) in _normalize(hit.text)
    m = hit.metadata
    if m.source != expected["source"]:
        return False
    if expected.get("chunk") is not None and m.chunk != expected["chunk"]:
        return False
    if expected.get("page") is not None:
        page = getattr(m, "page", None)
        if page is None:
            return False
        page_end = getattr(m, "pageEnd", None) or page
        if not page <= expected["page"] <= page_end:
            return False
    return True


def score_results(hits: Sequence[SearchResult], expected: Sequence[Expected]) -> Tuple[float, float]:
    """One question's (recall, reciprocal rank)."""
    found = sum(1 for e in expected if any(matches_expected(h, e) for h in hits))
    first = next(
        (i for i, h in enumerate(hits) if any(matches_expected(h, e) for e in expected)), None
    )
    return found / len(expected), 0.0 if first is None else 1 / (first + 1)


def evaluate(
    target: Any,
    questions: Sequence[EvalQuestion],
    *,
    top_k: int = 5,
    modes: Optional[Sequence[str]] = None,
    rerank: Optional[Any] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> EvalReport:
    """Run every question through each mode (and each mode with ``rerank``,
    as "<mode>+rerank") and report recall@k, MRR and hit rate."""
    modes = list(modes or ("vector", "keyword", "hybrid"))
    if not questions:
        raise ConfigError("No questions to evaluate")
    runs: List[Tuple[str, str, Any]] = [(m, m, False) for m in modes]
    if rerank:
        runs += [(f"{m}+rerank", m, rerank) for m in modes]

    results = []
    for name, mode, rr in runs:
        recall = mrr = 0.0
        missed = []
        for q in questions:
            opts: Dict[str, Any] = {"topK": top_k, "mode": mode, "rerank": rr}
            if filter:
                opts["filter"] = filter
            r, rank = score_results(target.search(q.question, opts), q.expected)
            recall += r
            mrr += rank
            if rank == 0:
                missed.append(q.question)
        n = len(questions)
        results.append(
            EvalRow(mode=name, recall=recall / n, mrr=mrr / n, hitRate=(n - len(missed)) / n, missed=missed)
        )
    return EvalReport(topK=top_k, questions=len(questions), results=results)


def _valid_expected(e: Any) -> bool:
    if isinstance(e, str):
        return bool(e.strip())
    if not isinstance(e, dict) or not isinstance(e.get("source"), str):
        return False
    return all(
        e.get(k) is None or (isinstance(e[k], int) and not isinstance(e[k], bool))
        for k in ("chunk", "page")
    )


def parse_questions(jsonl: str) -> List[EvalQuestion]:
    """Parse JSONL: one ``{"question": "...", "expected": [...]}`` per line."""
    questions = []
    for i, line in enumerate(jsonl.splitlines()):
        if not line.strip():
            continue
        where = f"line {i + 1}"
        try:
            q = json.loads(line)
        except ValueError:
            raise ConfigError(f"{where}: not valid JSON") from None
        if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ConfigError(f'{where}: "question" must be a non-empty string')
        expected = q.get("expected")
        if not isinstance(expected, list) or not expected:
            raise ConfigError(f'{where}: "expected" must be a non-empty array')
        if not all(_valid_expected(e) for e in expected):
            raise ConfigError(
                f'{where}: each "expected" item must be a string or {{ "source", "chunk"?, "page"? }}'
            )
        questions.append(EvalQuestion(question=q["question"], expected=expected))
    return questions


def format_report(report: EvalReport) -> str:
    """The report as an aligned text table."""
    header = ["mode", f"recall@{report.topK}", "MRR", "hit rate"]
    rows = [[r.mode, f"{r.recall:.3f}", f"{r.mrr:.3f}", f"{r.hitRate:.3f}"] for r in report.results]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def line(cells: List[str]) -> str:
        return "   ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)
        )

    noun = "question" if report.questions == 1 else "questions"
    return "\n".join([f"{report.questions} {noun}, top {report.topK}", "", line(header), *map(line, rows)])
