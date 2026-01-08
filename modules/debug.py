# modules/debug.py
import os
import cv2
import numpy as np

DEBUG_DIR = "debug"

def _ensure():
    os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# Basic saves
# -------------------------
def save_mask(name, mask):
    _ensure()
    cv2.imwrite(f"{DEBUG_DIR}/{name}_mask.png", mask)

def save_overlay(img, upper_mask, lower_mask):
    _ensure()
    overlay = img.copy()
    overlay[..., 2] = np.maximum(overlay[..., 2], upper_mask)  # R
    overlay[..., 1] = np.maximum(overlay[..., 1], lower_mask)  # G
    cv2.imwrite(f"{DEBUG_DIR}/overlay_RG.png", overlay)

def save_roi(name, roi):
    _ensure()
    cv2.imwrite(f"{DEBUG_DIR}/{name}_roi.png", roi)

# -------------------------
# Float visualizations
# -------------------------
def save_gray(name, x, autoscale=True):
    """
    x: float image (0~1 or signed)
    autoscale: True면 min/max 기준으로 강제 대비 확장
    """
    _ensure()
    v = x.copy()

    if autoscale:
        mn, mx = v.min(), v.max()
        if mx > mn:
            v = (v - mn) / (mx - mn)
        else:
            v = np.zeros_like(v)
    else:
        v = np.clip(v, 0, 1)

    v = np.clip(v * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{DEBUG_DIR}/{name}.png", v)

def save_signed(name, x, scale=5.0):
    """
    signed float 시각화 (0 = 중간 회색)
    """
    _ensure()
    v = 0.5 + x * scale
    v = np.clip(v, 0, 1)
    v = (v * 255).astype(np.uint8)
    cv2.imwrite(f"{DEBUG_DIR}/{name}.png", v)

def save_normal(name, N):
    """
    normal [-1,1] → RGB
    """
    _ensure()
    vis = (N * 0.5 + 0.5)
    vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{DEBUG_DIR}/{name}.png", vis)
