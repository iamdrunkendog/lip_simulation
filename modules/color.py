import cv2
import numpy as np


def apply_lip_color(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    r: int,
    g: int,
    b: int,
    opacity: float = 0.8,
):
    """
    img_bgr : BGR uint8 image
    mask    : lip mask (0~1 or 0~255)
    r, g, b : 0~255 integer
    opacity : 0.0 ~ 1.0
    """

    # --- mask normalize ---
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    mask_f = mask_f[..., None]

    # --- Target Color (RGB -> HSV) ---
    # Convert input RGB to BGR for OpenCV
    target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    
    target_h = float(target_hsv[0])  # 0~179
    target_s = float(target_hsv[1])  # 0~255

    # --- Source Image (BGR -> HSV) ---
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # We use uint8 for HSV replacement to stay within OpenCV's standard 0-179/0-255 ranges
    hsv_img_out = hsv_img.copy()
    hsv_img_out[..., 0] = int(target_h)
    hsv_img_out[..., 1] = int(target_s)
    # Value (hsv_img[..., 2]) is preserved from the original image

    # Convert back to BGR
    colored = cv2.cvtColor(hsv_img_out, cv2.COLOR_HSV2BGR)

    # --- blend only on lip mask ---
    out = (
        img_bgr.astype(np.float32) * (1.0 - mask_f * opacity)
        + colored.astype(np.float32) * (mask_f * opacity)
    )

    return np.clip(out, 0, 255).astype(np.uint8)
