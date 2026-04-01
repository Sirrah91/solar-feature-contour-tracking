import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from scr.utils.filesystem import is_empty

from scr.geometry.contours.extraction import find_contours
from scr.geometry.contours.area import contour_area

from scr.contours.expansion import extract_expanded_contour


def test_extract_expanded_contour() -> None:
    # Create a mock image: white circle (bright blob) in black background
    shape = (128, 128)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (64, 64)
    radius = 20
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    image = np.zeros(shape, dtype=float)
    image[mask] = 1000.0

    # Add some noise elsewhere
    noise = np.random.randn(*shape) * 10
    image += noise

    # Step 1: Find initial contour (simple thresholding)
    init_contours = find_contours(image, level=500)
    init_contours = sorted(init_contours, key=contour_area, reverse=True)
    if is_empty(init_contours):
        print("No initial contour found!")
        return

    init_contour = init_contours[0]

    # Step 2: Expand the contour
    expanded = extract_expanded_contour(
        image=image,
        contour=init_contour,
        expansion_threshold=500,
        iterations=2,
        return_mask=True
    )

    if expanded is None:
        print("Expansion failed.")
        return

    expanded_contour, expanded_mask = expanded

    # Step 3: Visualise results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image, cmap="gray")
    ax[0].plot(init_contour[:, 1], init_contour[:, 0], color="cyan", label="Initial Contour")
    ax[0].set_title("Input Image with Initial Contour")
    ax[0].legend()

    ax[1].imshow(expanded_mask, cmap="Reds", alpha=0.6)
    ax[1].plot(expanded_contour[:, 1], expanded_contour[:, 0], color="blue", label="Expanded Contour")
    ax[1].set_title("Expanded Region")
    ax[1].legend()

    plt.tight_layout()
    plt.show(block=False)
