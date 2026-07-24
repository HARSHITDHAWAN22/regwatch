"""
Chunking strategies (Strategy pattern).

Legal/regulatory text should NOT be split by fixed character count - that
cuts clauses mid-sentence and destroys retrieval quality. We split on
numbered section boundaries first, falling back to paragraph breaks.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    section_label: str | None
    char_start: int
    char_end: int


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        ...


class SectionBasedChunker(ChunkingStrategy):
    """
    Splits on patterns like '4.', '4.2', 'Para 4', 'Section 4:' at the
    start of a line - the way most RBI/SEBI circulars are actually structured.
    """
    SECTION_PATTERN = re.compile(
        r"(?m)^(?:Para(?:graph)?\s+)?(\d+(?:\.\d+)*)[\.\):]?\s+"
    )

    def chunk(self, text: str) -> list[Chunk]:
        matches = list(self.SECTION_PATTERN.finditer(text))
        if not matches:
            return ParagraphChunker().chunk(text)

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_label = match.group(1)
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 20:  # skip near-empty fragments
                chunks.append(Chunk(
                    text=chunk_text,
                    section_label=section_label,
                    char_start=start,
                    char_end=end,
                ))
        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Fallback: split on blank-line-separated paragraphs."""

    def chunk(self, text: str) -> list[Chunk]:
        chunks = []
        offset = 0
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) > 20:
                start = text.find(para, offset)
                end = start + len(para)
                chunks.append(Chunk(text=para, section_label=None, char_start=start, char_end=end))
                offset = end
        return chunks


def get_chunker(strategy: str = "section") -> ChunkingStrategy:
    if strategy == "section":
        return SectionBasedChunker()
    if strategy == "paragraph":
        return ParagraphChunker()
    raise ValueError(f"Unknown chunking strategy: {strategy}")
