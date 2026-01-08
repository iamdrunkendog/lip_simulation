import cv2
import numpy as np


def apply_lip_color(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    hue_shift=0,
    saturation=1.0,
    value=1.0,
    opacity=0.8,
):
    """
    img_bgr : BGR uint8 image
    mask    : lip mask (0~1 or 0~255)
    """

    # --- mask normalize ---
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    mask_f = mask_f[..., None]

    # --- BGR -> HSV ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue shift (OpenCV HSV: 0~179)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    # Saturation / Value
    hsv[..., 1] *= saturation
    hsv[..., 2] *= value

    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)

    colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- blend only on lip mask ---
    out = (
        img_bgr * (1.0 - mask_f * opacity)
        + colored * (mask_f * opacity)
    )

    return out.astype(np.uint8)
