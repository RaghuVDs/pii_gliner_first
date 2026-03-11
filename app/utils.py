def overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Returns True if two text spans overlap."""
    return max(start1, start2) < min(end1, end2)