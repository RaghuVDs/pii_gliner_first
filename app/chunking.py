from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[Tuple[int, int, str]]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append((start, end, text[start:end]))
        if end == n:
            break
        start = max(start + 1, end - overlap)
    return chunks