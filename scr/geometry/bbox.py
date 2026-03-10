def bboxes_intersect(
        b1: tuple[float, float, float, float],
        b2: tuple[float, float, float, float],
) -> bool:
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2
    return not (
        maxx1 < minx2 or
        maxx2 < minx1 or
        maxy1 < miny2 or
        maxy2 < miny1
    )
