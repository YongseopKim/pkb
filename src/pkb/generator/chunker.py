"""Text chunking for vector storage."""

import re


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into chunks respecting sentence boundaries.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    # Split into sentences (handles Korean/English mixed text)
    sentences = re.split(r'(?<=[.!?。\n])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return [text.strip()]

    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Calculate overlap: keep trailing sentences that fit in overlap
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s) + 1

            current_chunk = overlap_chunk
            current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            if current_len < 0:
                current_len = 0

        current_chunk.append(sentence)
        current_len += sentence_len + (1 if current_len > 0 else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_chunks_for_chromadb(
    chunks: list[str], metadata: dict
) -> list[dict]:
    """Prepare chunk data in ChromaDB-ready format.

    Args:
        chunks: List of text chunks.
        metadata: Base metadata to attach to each chunk.

    Returns:
        List of dicts with id, document, metadata keys.
    """
    bundle_id = metadata.get("bundle_id", "unknown")
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "id": f"{bundle_id}-chunk-{i}",
            "document": chunk,
            "metadata": {**metadata},
        })
    return result
