import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.chunker import get_chunker, SectionBasedChunker, ParagraphChunker


def test_section_chunker_splits_numbered_paragraphs():
    text = """1. This is the first clause about KYC limits.

2. This is the second clause about UPI transaction caps.

2.1 This is a sub-clause with more detail.
"""
    chunker = get_chunker("section")
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert chunks[0].section_label == "1"
    assert "KYC" in chunks[0].text
    assert chunks[2].section_label == "2.1"


def test_section_chunker_falls_back_to_paragraphs_when_no_numbering():
    text = "First paragraph with no numbering at all here.\n\nSecond paragraph, also unnumbered."
    chunker = get_chunker("section")
    chunks = chunker.chunk(text)
    assert len(chunks) == 2


def test_paragraph_chunker_skips_short_fragments():
    text = "A real paragraph with enough content to be a valid chunk here.\n\nHi\n\nAnother real paragraph with sufficient length to pass the filter."
    chunker = ParagraphChunker()
    chunks = chunker.chunk(text)
    assert all(len(c.text) > 20 for c in chunks)
    assert len(chunks) == 2


def test_invalid_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        get_chunker("nonexistent")
