import cv2
import numpy as np

def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def gaussian_blur(img_bgr: np.ndarray, ksize: int = 9) -> np.ndarray:
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img_bgr, (k, k), 0)

def edge_canny(img_bgr: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=int(t1), threshold2=int(t2))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def adjust_brightness_contrast(img_bgr: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """
    brightness: -100..100
    contrast:   -100..100
    """
    b = int(np.clip(brightness, -100, 100))
    c = int(np.clip(contrast, -100, 100))

    # Contrast factor
    if c >= 0:
        alpha = 1 + (c / 100.0) * 2.0   # up to ~3.0
    else:
        alpha = 1 + (c / 100.0)        # down to ~0.0+

    beta = b * 2.55  # map -100..100 to -255..255-ish

    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out

def rotate_90(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

def flip_horizontal(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(img_bgr, 1)
