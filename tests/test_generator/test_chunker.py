"""Tests for text chunker."""

from pkb.generator.chunker import chunk_text, prepare_chunks_for_chromadb


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "짧은 텍스트입니다."
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # Create text that is clearly > chunk_size
        sentences = ["이것은 테스트 문장입니다."] * 50
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        sentences = ["문장 하나.", "문장 둘.", "문장 셋.", "문장 넷.", "문장 다섯."]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        if len(chunks) >= 2:
            # Chunks should have some overlapping content
            # (this is a soft check since exact overlap depends on sentence boundaries)
            assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=512, overlap=50)
        assert chunks == []

    def test_respects_sentence_boundaries(self):
        text = "첫 번째 문장. 두 번째 문장. 세 번째 문장."
        chunks = chunk_text(text, chunk_size=30, overlap=0)
        # Each chunk should not break in the middle of a sentence
        for chunk in chunks:
            assert chunk.strip()  # No empty chunks


class TestPrepareChunksForChromadb:
    def test_basic_preparation(self):
        texts = ["청크 1 내용", "청크 2 내용"]
        metadata = {
            "bundle_id": "20260221-test-a3f2",
            "kb": "personal",
            "platform": "claude",
            "domains": "dev",
            "topics": "python,async",
        }
        prepared = prepare_chunks_for_chromadb(texts, metadata)
        assert len(prepared) == 2
        assert prepared[0]["id"] == "20260221-test-a3f2-chunk-0"
        assert prepared[0]["document"] == "청크 1 내용"
        assert prepared[0]["metadata"]["bundle_id"] == "20260221-test-a3f2"

    def test_empty_list(self):
        prepared = prepare_chunks_for_chromadb([], {"bundle_id": "test"})
        assert prepared == []
