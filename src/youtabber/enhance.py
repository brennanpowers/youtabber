import cv2
import numpy as np

FACTOR = 2
# Unsharp mask: add back the difference between the image and a blurred copy to crisp up edges
SHARPEN_AMOUNT = 1.0
SHARPEN_SIGMA = 1.5
# Levels: anything darker than BLACK becomes black and lighter than WHITE becomes white.
# Grays in between stay gray, which keeps light staff lines and measure numbers visible.
BLACK = 80
WHITE = 230


def enhance(image: np.ndarray) -> np.ndarray:
    """Upscale whitened notation and sharpen its edges for printing."""
    up = cv2.resize(image, None, fx=FACTOR, fy=FACTOR, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    sharp = up + SHARPEN_AMOUNT * (up - cv2.GaussianBlur(up, (0, 0), SHARPEN_SIGMA))
    return np.clip((sharp - BLACK) * 255 / (WHITE - BLACK), 0, 255).astype(np.uint8)
