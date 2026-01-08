import cv2
import numpy as np

def make_liquid_mask(lip01, erosion_px=6, blur_sigma=10):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion_px * 2 + 1, erosion_px * 2 + 1)
    )
    eroded = cv2.erode(lip01, kernel)
    liquid = cv2.GaussianBlur(eroded, (0, 0), blur_sigma)
    return np.clip(liquid, 0, 1).astype(np.float32)


def generate_normal_from_height(height):
    height = height.astype(np.float32)
    dHx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    dHy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)

    Nx = -dHx
    Ny = -dHy
    Nz = np.ones_like(Nx)

    N = np.dstack([Nx, Ny, Nz])
    N /= (np.linalg.norm(N, axis=2, keepdims=True) + 1e-6)
    return N


def build_layered_fake_normal(
    roi, lip01_roi,
    liquid_blur_sigma=18,
    height_gain=8.0,
    high_blur_sigma=2.0,
    high_gain=3.0,
    alpha=0.25
):
    # LOW
    liquid_mask = make_liquid_mask(lip01_roi)
    height_low = cv2.GaussianBlur(
        liquid_mask * height_gain, (0, 0), liquid_blur_sigma
    )
    N_low = generate_normal_from_height(height_low)

    # HIGH
    gray = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    low_gray = cv2.GaussianBlur(gray, (0, 0), high_blur_sigma)
    high = (gray - low_gray) * (lip01_roi > 0)

    N_high = generate_normal_from_height(high * high_gain)

    # MERGE
    N = N_low + alpha * N_high
    N /= (np.linalg.norm(N, axis=2, keepdims=True) + 1e-6)
    return N
