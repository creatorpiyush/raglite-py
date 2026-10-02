import re
import unicodedata
from typing import Iterator, List, Tuple

from .scripts import is_bigram_char, is_mark

# Name of the keyword tokenizer, recorded with every keyword index. Change it
# whenever ``tokenize`` output changes, so existing keyword indexes are rebuilt.
TOKENIZER_NAME = "raglite-v1"

# Invisible characters removed before tokenizing: soft hyphen, ZWNJ, ZWJ, word joiner.
_IGNORED = re.compile("[­‌‍⁠]")
_JOINERS = frozenset("_-.")


def tokenize(text: str) -> List[str]:
    """Split text into keyword search terms.

    Must produce exactly the same terms as the TypeScript SDK; the shared
    fixtures in tests/fixtures/shared check this.

    1. NFKC normalisation, then lowercase.
    2. Runs of letters, digits and combining marks are words. ``_``, ``-`` and
       ``.`` between word characters join them into one term (``gpt-4.1``,
       ``err_42``), and the joined parts are emitted as terms too.
    3. Han, kana, Hangul, Thai, Lao, Khmer and Myanmar runs are emitted as
       overlapping character bigrams, or a unigram for a single character.
       This needs no dictionary and behaves the same in both SDKs.
    """
    normalized = _IGNORED.sub("", unicodedata.normalize("NFKC", text).lower())
    tokens: List[str] = []

    compound = ""
    part = ""
    parts: List[str] = []
    joiner = ""
    bigram_run: List[str] = []

    def end_word() -> None:
        nonlocal compound, part, parts, joiner
        if compound:
            parts.append(part)
            tokens.append(compound)
            if len(parts) > 1:
                tokens.extend(parts)
        compound = ""
        part = ""
        parts = []
        joiner = ""

    def end_bigram_run() -> None:
        nonlocal bigram_run
        if len(bigram_run) == 1:
            tokens.append(bigram_run[0])
        else:
            for i in range(len(bigram_run) - 1):
                tokens.append(bigram_run[i] + bigram_run[i + 1])
        bigram_run = []

    for cluster, kind in _clusters(normalized):
        if kind == "bigram":
            end_word()
            bigram_run.append(cluster)
        elif kind == "word":
            end_bigram_run()
            if joiner:
                parts.append(part)
                part = ""
                compound += joiner
                joiner = ""
            part += cluster
            compound += cluster
        elif kind == "joiner":
            end_bigram_run()
            if compound and not joiner:
                joiner = cluster
            else:
                end_word()
        else:
            end_word()
            end_bigram_run()
    end_word()
    end_bigram_run()
    return tokens


def _clusters(text: str) -> Iterator[Tuple[str, str]]:
    """Group each character with the combining marks after it and classify the group."""
    current = ""
    kind = "other"
    for ch in text:
        if current and is_mark(ch):
            current += ch
            continue
        if current:
            yield current, kind
        current = ch
        kind = _classify(ch)
    if current:
        yield current, kind


def _classify(ch: str) -> str:
    if unicodedata.category(ch)[0] in "LNM":
        return "bigram" if is_bigram_char(ord(ch)) else "word"
    return "joiner" if ch in _JOINERS else "other"
