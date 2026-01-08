import cv2
import numpy as np
from PIL import Image, ExifTags

from modules.lip_landmarks import get_lip_masks
from modules.soft_mask import combine_upper_lower
from modules.fake_normal import build_layered_fake_normal
from modules.lcs import compute_lip_coordinate_system
from modules.specular import compute_specular, screen
from modules.clearcoat import build_clearcoat_weight, apply_clearcoat
from modules.color import apply_lip_color


# =====================================================
# EXIF SAFE IMAGE LOADER
# =====================================================
def load_image_exif_safe(path):
    img_pil = Image.open(path)

    try:
        exif = img_pil._getexif()
        if exif is not None:
            orientation_key = None
            for k, v in ExifTags.TAGS.items():
                if v == "Orientation":
                    orientation_key = k
                    break

            if orientation_key is not None:
                orientation = exif.get(orientation_key, None)
                if orientation == 3:
                    img_pil = img_pil.rotate(180, expand=True)
                elif orientation == 6:
                    img_pil = img_pil.rotate(270, expand=True)
                elif orientation == 8:
                    img_pil = img_pil.rotate(90, expand=True)
    except Exception:
        pass

    img = np.array(img_pil)
    if img.ndim == 3:
        img = img[:, :, ::-1].copy()  # RGB → BGR
    return img


# =====================================================
# RENDER ONCE
# =====================================================
def render_lips(image_path, params):
    # ------------------
    # Load image
    # ------------------
    img = load_image_exif_safe(image_path)
    H, W = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    # ------------------
    # Lip masks
    # ------------------
    upper_mask, lower_mask, lip_mask, _, _ = get_lip_masks(img)

    lip01 = combine_upper_lower(
        upper_mask,
        lower_mask,
        edge_width_upper=params["EDGE_UPPER"],
        edge_width_lower=params["EDGE_LOWER"]
    ).astype(np.float32)

    
    # ------------------
    # Lip color
    # ------------------
    color_params = params.get("COLOR", {})

    img_colored = apply_lip_color(
        img,
        lip01, # Use soft mask for feathering
        r=color_params.get("R", 0),
        g=color_params.get("G", 0),
        b=color_params.get("B", 0),
        opacity=color_params.get("OPACITY", 0.8),
        mode=params.get("BLENDING_MODE", "normal")
    )




    # ------------------
    # ROI
    # ------------------
    ys, xs = np.where(lip_mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    pad = 10
    y0 = max(0, y0 - pad)
    y1 = min(H - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(W - 1, x1 + pad)

    img_colored_f = img_colored.astype(np.float32) / 255.0
    roi = img_colored_f[y0:y1+1, x0:x1+1]
    lip01_roi = lip01[y0:y1+1, x0:x1+1]
    upper_roi = upper_mask[y0:y1+1, x0:x1+1]
    lower_roi = lower_mask[y0:y1+1, x0:x1+1]

    # ------------------
    # Fake normal
    # ------------------
    N = build_layered_fake_normal(
        roi,
        lip01_roi,
        liquid_blur_sigma=params["LIQUID_BLUR_SIGMA"],
        height_gain=params["HEIGHT_GAIN"],
        high_blur_sigma=params["HIGH_BLUR_SIGMA"],
        high_gain=params["HIGH_GAIN"],
        alpha=params["ALPHA"]
    )

    # -------------------
    # Specular
    # ------------------
    spec = compute_specular(
        N,
        lip01_roi,
        params["LIGHT_DIR"],
        strength=params["SPEC_STRENGTH"],
        shininess=params["SHININESS"]
    )

    out_roi = screen(roi, spec[..., None])

    # ------------------
    # LCS + Clearcoat
    # ------------------
    x_hat, y_hat, r = compute_lip_coordinate_system(
        upper_roi,
        lower_roi,
        params["MIDLINE_BIAS"]
    )

    HWF = build_clearcoat_weight(
        x_hat, y_hat, r,
        (upper_roi > 0).astype(np.float32),
        (lower_roi > 0).astype(np.float32),
        lip01_roi,
        lower_center=params["LOWER_CENTER"],
        upper_center=params["UPPER_CENTER"]
    )

    clear = apply_clearcoat(
        N[..., 2],
        HWF,
        strength=params["CLEAR_STRENGTH"],
        exponent=params["CLEAR_EXPONENT"]
    )

    out_roi = screen(out_roi, clear[..., None])

    # ------------------
    # Composite
    # ------------------
    # Use img_colored as the base for the face instead of img_f
    # so that the chosen lipstick color is actually applied.
    face = img_colored.astype(np.float32) / 255.0
    
    # Add specular and clearcoat highlights (delta) only to the lip area
    delta = (out_roi - roi) * (lip01_roi > 0)[..., None]
    face[y0:y1+1, x0:x1+1] += delta
    face = np.clip(face, 0, 1)

    # ------------------
    # Return all debug
    # ------------------
    return {
        "original": img,
        "final": (face * 255).astype(np.uint8),

        "original_roi": img[y0:y1+1, x0:x1+1],
        "final_roi": (face[y0:y1+1, x0:x1+1] * 255).astype(np.uint8),

        "lip01": (lip01_roi * 255).astype(np.uint8),

        "normal_Nx": ((N[..., 0] * 0.5 + 0.5) * 255).astype(np.uint8),
        "normal_Ny": ((N[..., 1] * 0.5 + 0.5) * 255).astype(np.uint8),
        "normal_Nz": (N[..., 2] * 255).astype(np.uint8),

        "normal_length": (
            np.clip(np.sqrt((N ** 2).sum(axis=-1)), 0, 1) * 255
        ).astype(np.uint8),

        "x_hat": ((x_hat + 1) * 0.5 * 255).astype(np.uint8),
        "y_hat": ((y_hat + 1) * 0.5 * 255).astype(np.uint8),
        "r": (r * 255).astype(np.uint8),

        "hwf": (HWF * 255).astype(np.uint8),
        "spec": (spec * 255).astype(np.uint8),
        "clear_weight": (HWF * 255).astype(np.uint8),
        "clear_spec": (clear * 255).astype(np.uint8),
    }
