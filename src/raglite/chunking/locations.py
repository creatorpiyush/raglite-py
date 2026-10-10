"""Page and markdown section of each chunk, for citations. Mirrors locations.ts."""

import re
from bisect import bisect_right
from typing import Dict, List, Optional, TypedDict, Union

from .recursive import Span, count_units


class Heading(TypedDict):
    start: int
    section: str


class SourceLayout(TypedDict, total=False):
    # Unit offset at which each page starts.
    pageStarts: List[int]
    # Unit offset and heading path of each markdown heading, in order.
    headings: List[Heading]


def page_starts(pages: List[str]) -> List[int]:
    """Page start offsets for text that is the pages joined with whitespace."""
    starts: List[int] = []
    offset = 0
    for page in pages:
        starts.append(offset)
        offset += count_units(page)
    return starts


_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_LINE_BREAK = re.compile(r"\r\n|\n|\r")


def markdown_headings(text: str) -> List[Heading]:
    """ATX headings outside fenced code blocks, each with its full heading path."""
    headings: List[Heading] = []
    # Open headings, outermost first; a heading closes those at its level or deeper.
    path: List[tuple] = []
    offset = 0
    fence: Optional[str] = None
    for line in _LINE_BREAK.split(text):
        fence_match = _FENCE.match(line)
        marker = fence_match.group(1) if fence_match else None
        if marker and (fence is None or fence == marker):
            fence = marker if fence is None else None
        elif fence is None:
            match = _HEADING.match(line)
            if match:
                level = len(match.group(1))
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, match.group(2).strip()))
                headings.append({"start": offset, "section": " > ".join(t for _, t in path)})
        offset += count_units(line)
    return headings


def chunk_location(layout: SourceLayout, span: Span) -> Dict[str, Union[int, str]]:
    """Page range and section of a chunk. The section is the heading in effect
    where the chunk starts, or else the first heading inside the chunk.
    Keys are present only when known: page, pageEnd (if > page), section."""
    location: Dict[str, Union[int, str]] = {}
    starts = layout.get("pageStarts")
    if starts:
        page = bisect_right(starts, span.start)
        page_end = bisect_right(starts, span.end - 1)
        location["page"] = page
        if page_end > page:
            location["pageEnd"] = page_end
    headings = layout.get("headings")
    if headings:
        i = bisect_right([h["start"] for h in headings], span.start) - 1
        heading = headings[i] if i >= 0 else next((h for h in headings if h["start"] < span.end), None)
        if heading:
            location["section"] = heading["section"]
    return location
