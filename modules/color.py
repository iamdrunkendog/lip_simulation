import cv2
import numpy as np


def apply_lip_color(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    r: int,
    g: int,
    b: int,
    opacity: float = 0.8,
    mode: str = "normal",
    base_desat: float = 0.0,
    v_weight: float = 0.0
):
    """
    img_bgr    : BGR uint8 image
    mask       : lip mask (0~1 or 0~255)
    r, g, b    : 0~255 integer
    opacity    : 0.0 ~ 1.0
    mode       : "normal" or "softlight"
    base_desat : 0.0 ~ 1.0 (0.0: keep original, 1.0: fully grayscale base)
    v_weight   : 0.0 ~ 1.0 (명도 반영 비중. 1.0이면 대상 색상의 명도를 그대로 사용)
    """

    # --- mask normalize ---
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    mask_f = mask_f[..., None]

    # --- Pre-processing: Base Desaturation ---
    # We apply this to a copy of the original image to use as the base for coloring
    if base_desat > 0:
        hsv_base = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Reduce saturation: S = S * (1 - base_desat)
        hsv_base[..., 1] *= (1.0 - base_desat)
        img_base_processed = cv2.cvtColor(hsv_base.astype(np.uint8), cv2.COLOR_HSV2BGR)
    else:
        img_base_processed = img_bgr

    if mode == "softlight":
        # --- Soft Light Blending ---
        base = img_base_processed.astype(np.float32) / 255.0
        blend = np.array([b, g, r], dtype=np.float32) / 255.0 # Target BGR
        
        # broadcast blend color to shape
        Cs = blend[None, None, :]
        Cb = base
        
        # D(Cb)
        D_Cb = np.where(Cb <= 0.25, ((16 * Cb - 12) * Cb + 4) * Cb, np.sqrt(Cb))
        
        # B(Cb, Cs)
        res = np.where(Cs <= 0.5, 
                       Cb - (1.0 - 2.0 * Cs) * Cb * (1.0 - Cb),
                       Cb + (2.0 * Cs - 1.0) * (D_Cb - Cb))
        
        colored = (np.clip(res, 0, 1) * 255).astype(np.uint8)

    else:
        # --- Normal Mode (Hue/Sat Replacement) ---
        target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        
        target_h = float(target_hsv[0])
        target_s = float(target_hsv[1])

        # Source for replacement is also the (potentially desaturated) base
        hsv_img = cv2.cvtColor(img_base_processed, cv2.COLOR_BGR2HSV)
        
        hsv_img_out = hsv_img.copy()
        hsv_img_out[..., 0] = int(target_h)
        hsv_img_out[..., 1] = int(target_s)

        colored = cv2.cvtColor(hsv_img_out, cv2.COLOR_HSV2BGR)

    # --- Value (Brightness) adjustment based on v_weight ---
    if v_weight > 0:
        target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
        target_v = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0, 2]
        
        hsv_colored = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV).astype(np.float32)
        base_v = hsv_colored[..., 2]
        
        # V_final = V_base * (1 - w) + V_target * w
        new_v = base_v * (1.0 - v_weight) + target_v * v_weight
        hsv_colored[..., 2] = np.clip(new_v, 0, 255)
        
        colored = cv2.cvtColor(hsv_colored.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- blend only on lip mask ---
    # Important: The background outside opacity should be original img_bgr,
    # but the part being colored is based on img_base_processed.
    # However, usually we want to see the desaturation effect as part of the simulation.
    out = (
        img_bgr.astype(np.float32) * (1.0 - mask_f * opacity)
        + colored.astype(np.float32) * (mask_f * opacity)
    )

    return np.clip(out, 0, 255).astype(np.uint8)
