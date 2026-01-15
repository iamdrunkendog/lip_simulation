import cv2
import numpy as np
from PIL import Image, ExifTags

from modules.lip_landmarks import get_lip_masks
from modules.soft_mask import combine_upper_lower
from modules.fake_normal import build_layered_fake_normal
from modules.lcs import compute_lip_coordinate_system
from modules.specular import compute_specular, screen, blend_specular
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

    img_pil = img_pil.convert("RGB")
    img = np.array(img_pil)
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
        mode=params.get("BLENDING_MODE", "normal"),
        base_desat=params.get("BASE_DESATURATION", 0.0),
        v_weight=params.get("VALUE_WEIGHT", 0.0),
        color_depth=params.get("DEEP_COLOR", 0.0)
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
    N_dict = build_layered_fake_normal(
        roi,
        lip01_roi,
        liquid_blur_sigma=params["LIQUID_BLUR_SIGMA"],
        height_gain=params["HEIGHT_GAIN"],
        high_blur_sigma=params["HIGH_BLUR_SIGMA"],
        high_gain=params["HIGH_GAIN"],
        alpha=params["ALPHA"],
        surface_smoothing=params.get("TEXTURE_SMOOTHING", 0.0)
    )
    N = N_dict["final"]
    N_low = N_dict["low"]
    N_high = N_dict["high"]

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

    # [NEW] Colored Specular Logic
    # Multiply the white specular map by the base image (roi) to pick up the lip color.
    # This ensures "white blend" issues are minimized.
    # Fix broadcasting error: spec is (H,W), roi is (H,W,3) -> need spec[..., None]
    spec_colored = spec[..., None] * roi 

    out_roi = blend_specular(
        roi, spec_colored,
        mode=params.get("SPEC_BLEND_MODE", "screen")
    )

    # [Comparison] White Specular Version
    out_roi_white = blend_specular(
        roi, spec,
        mode=params.get("SPEC_BLEND_MODE", "screen")
    )

    # ------------------
    # LCS (Coordinate System)
    # ------------------
    x_hat, y_hat, r = compute_lip_coordinate_system(
        upper_roi,
        lower_roi,
        params["MIDLINE_BIAS"]
    )

    # ------------------
    # Clearcoat Shape (HWF)
    # ------------------
    HWF = build_clearcoat_weight(
        x_hat, y_hat, r,
        (upper_roi > 0).astype(np.float32),
        (lower_roi > 0).astype(np.float32),
        lip01_roi,
        lower_center=params["LOWER_CENTER"],
        upper_center=params["UPPER_CENTER"],
        lower_sigma_y=params.get("LOWER_WIDTH", 0.35),
        upper_sigma_y=params.get("UPPER_WIDTH", 0.25),
        cupid_boost=params.get("CUPID_BOOST", 0.0)
    )

    # ------------------
    # Clearcoat Apply
    # ------------------
    # Custom Normal for Clear Coat to preserve texture while flattening shape
    influence = params.get("CLEAR_NORMAL_INFLUENCE", 1.0)
    
    # Flatten ONLY the macro shape (N_low) based on influence
    # 1.0 -> N_low (Curved), 0.0 -> Flat (0,0,1)
    N_low_flat = N_low * influence + np.array([0, 0, 1]) * (1.0 - influence)
    N_low_flat /= (np.linalg.norm(N_low_flat, axis=2, keepdims=True) + 1e-6)
    
    # Re-apply the texture detail (N_high) with the same alpha
    N_clear = N_low_flat + params["ALPHA"] * N_high
    N_clear /= (np.linalg.norm(N_clear, axis=2, keepdims=True) + 1e-6)

    # [NEW] Clear Coat Light Direction Control (X-axis tilt)
    light_x = params.get("CLEAR_LIGHT_X", 0.0)
    
    # Virtual Lightsource Vector L = normalize([light_x, 0, 1])
    # We want dot(N, L). Since Ly=0, dot = Nx*Lx + Nz*Lz
    L_vec = np.array([light_x, 0.0, 1.0], dtype=np.float32)
    L_vec /= np.linalg.norm(L_vec)
    
    # Calculate Dot Product
    # N_clear[..., 0] is Nx, N_clear[..., 2] is Nz
    # Clear_Scalar = Nx * Lx + Nz * Lz
    clear_scalar = N_clear[..., 0] * L_vec[0] + N_clear[..., 2] * L_vec[2]

    clear = apply_clearcoat(
        clear_scalar,
        HWF,
        strength=params["CLEAR_STRENGTH"],
        exponent=params["CLEAR_EXPONENT"]
    )

    out_roi = screen(out_roi, clear[..., None])
    out_roi_white = screen(out_roi_white, clear[..., None])

    # ------------------
    # Composite
    # ------------------
    # Use img_colored as the base for the face instead of img_f
    # so that the chosen lipstick color is actually applied.
    face = img_colored.astype(np.float32) / 255.0
    
    # Add specular and clearcoat highlights (delta) only to the lip area
    mask_roi = (lip01_roi > 0)[..., None]
    
    delta = (out_roi - roi) * mask_roi
    face[y0:y1+1, x0:x1+1] += delta
    face = np.clip(face, 0, 1)

    # Calculate final ROI for white spec comparison (blending onto face ROI)
    delta_white = (out_roi_white - roi) * mask_roi
    face_roi_white = roi + delta_white
    face_roi_white = np.clip(face_roi_white, 0, 1)

    # ------------------
    # Return all debug
    # ------------------
    return {
        "original": img,
        "final": (face * 255).astype(np.uint8),

        "original_roi": img[y0:y1+1, x0:x1+1],
        "final_roi": (face[y0:y1+1, x0:x1+1] * 255).astype(np.uint8),
        "final_roi_white": (face_roi_white * 255).astype(np.uint8),

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
        "spec_colored": (spec_colored * 255).astype(np.uint8),
        "clear_weight": (HWF * 255).astype(np.uint8),
        "clear_spec": (clear * 255).astype(np.uint8),
    }
