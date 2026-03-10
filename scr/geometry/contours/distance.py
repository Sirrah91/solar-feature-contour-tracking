from scipy.spatial import distance

from scr.utils.types_alias import Contour


def contours_distance(
        contour1: Contour,
        contour2: Contour
) -> float:
    return distance.cdist(contour1, contour2, "euclidean").min()
