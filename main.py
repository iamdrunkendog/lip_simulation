import cv2
import numpy as np

from modules.lip_landmarks import get_lip_masks
from modules.soft_mask import combine_upper_lower

from modules.fake_normal import build_layered_fake_normal
from modules.lcs import compute_lip_coordinate_system
from modules.specular import compute_specular, screen
from modules.clearcoat import build_clearcoat_weight, apply_clearcoat

# =====================================================
# CONFIG (PIPELINE ONLY)
# =====================================================
IMAGE_PATH = "test.jpg"

EDGE_UPPER = 6
EDGE_LOWER = 8
MIDLINE_BIAS = 0.12

# ---- fake normal (BASELINE: DO NOT TOUCH)
LIQUID_BLUR_SIGMA = 18
HEIGHT_GAIN = 8.0
HIGH_BLUR_SIGMA = 2.0
HIGH_GAIN = 3.0
ALPHA = 0.25

# ---- spec (FIXED)
SPEC_STRENGTH = 0.18
SHININESS = 60
LIGHT_DIR = np.array([0.0, -0.25, 1.0], np.float32)
LIGHT_DIR /= (np.linalg.norm(LIGHT_DIR) + 1e-6)

# ---- clear coat (FIXED PRESET: GLOSS)
CLEAR_STRENGTH = 0.4
CLEAR_EXPONENT = 30

# =====================================================
# LOAD IMAGE
# =====================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Failed to load image")

H, W = img.shape[:2]
img_f = img.astype(np.float32) / 255.0

# =====================================================
# 1. LIP MASKS (FROZEN)
# =====================================================
upper_mask, lower_mask, lip_mask, _, _ = get_lip_masks(img)

lip01 = combine_upper_lower(
    upper_mask,
    lower_mask,
    edge_width_upper=EDGE_UPPER,
    edge_width_lower=EDGE_LOWER
).astype(np.float32)

# =====================================================
# 2. ROI
# =====================================================
ys, xs = np.where(lip_mask > 0)
y0, y1 = ys.min(), ys.max()
x0, x1 = xs.min(), xs.max()

pad = 10
y0 = max(0, y0 - pad)
y1 = min(H - 1, y1 + pad)
x0 = max(0, x0 - pad)
x1 = min(W - 1, x1 + pad)

roi = img_f[y0:y1+1, x0:x1+1]
lip01_roi = lip01[y0:y1+1, x0:x1+1]
upper_roi = upper_mask[y0:y1+1, x0:x1+1]
lower_roi = lower_mask[y0:y1+1, x0:x1+1]

# =====================================================
# 3. FAKE NORMAL (LOW + HIGH)
# =====================================================
N = build_layered_fake_normal(
    roi,
    lip01_roi,
    liquid_blur_sigma=LIQUID_BLUR_SIGMA,
    height_gain=HEIGHT_GAIN,
    high_blur_sigma=HIGH_BLUR_SIGMA,
    high_gain=HIGH_GAIN,
    alpha=ALPHA
)

# =====================================================
# 4. SPECULAR
# =====================================================
spec = compute_specular(
    N,
    lip01_roi,
    LIGHT_DIR,
    strength=SPEC_STRENGTH,
    shininess=SHININESS
)

out_roi = screen(roi, spec[..., None])

# =====================================================
# 5. LIP COORDINATE SYSTEM
# =====================================================
x_hat, y_hat, r = compute_lip_coordinate_system(
    upper_roi,
    lower_roi,
    midline_bias=MIDLINE_BIAS
)

# =====================================================
# 6. CLEAR COAT (HWF + APPLY)
# =====================================================
HWF = build_clearcoat_weight(
    x_hat,
    y_hat,
    r,
    (upper_roi > 0).astype(np.float32),
    (lower_roi > 0).astype(np.float32),
    lip01_roi
)

clear = apply_clearcoat(
    N[..., 2],   # Nz
    HWF,
    strength=CLEAR_STRENGTH,
    exponent=CLEAR_EXPONENT
)

out_roi = screen(out_roi, clear[..., None])

# =====================================================
# 7. COMPOSITE BACK TO FACE
# =====================================================
face = img_f.copy()
delta = (out_roi - roi) * (lip01_roi > 0)[..., None]
face[y0:y1+1, x0:x1+1] += delta
face = np.clip(face, 0, 1)

# =====================================================
# 8. OUTPUT
# =====================================================
cv2.imwrite(
    "debug/final_face_result.png",
    (face * 255).astype(np.uint8)
)

print("DONE - BASELINE GLOSS PIPELINE")
