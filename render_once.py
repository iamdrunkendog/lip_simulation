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

    # ------------------
    # Base Lip Mask (Strict)
    # ------------------
    lip01 = combine_upper_lower(
        upper_mask,
        lower_mask,
        edge_width_upper=params["EDGE_UPPER"],
        edge_width_lower=params["EDGE_LOWER"]
    ).astype(np.float32)

    # ------------------
    # Mask Expansion (Dual Flow)
    # ------------------
    expand_ratio = params.get("CLEAR_EXPAND_RATIO", 0.0)
    expand_px = 0
    
    if expand_ratio > 0.0:
        # Calculate BBox of the full lip
        # Combine upper and lower to get full lip bbox
        full_mask = cv2.bitwise_or(upper_mask, lower_mask)
        ys_b, xs_b = np.where(full_mask > 0)
        
        if len(ys_b) > 0:
            h_lip = ys_b.max() - ys_b.min()
            w_lip = xs_b.max() - xs_b.min()
            ref_size = max(h_lip, w_lip)
            expand_px = int(ref_size * expand_ratio / 100.0)
    
    # lip01 (Strict) -> For Color & Specular
    # lip01_dilated (Expanded) -> For Clear Coat & Fake Normal (Shape)
    
    if expand_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Dilate binary masks first
        upper_dilated = cv2.dilate(upper_mask, kernel, iterations=expand_px)
        lower_dilated = cv2.dilate(lower_mask, kernel, iterations=expand_px)
        
        # Soft mask for expanded area
        lip01_dilated = combine_upper_lower(
            upper_dilated,
            lower_dilated,
            edge_width_upper=params["EDGE_UPPER"],
            edge_width_lower=params["EDGE_LOWER"]
        ).astype(np.float32)
        
        # Mask for ROI slicing later
        upper_mask_for_shape = upper_dilated
        lower_mask_for_shape = lower_dilated
        lip01_for_shape = lip01_dilated
    else:
        lip01_dilated = lip01
        upper_mask_for_shape = upper_mask
        lower_mask_for_shape = lower_mask
        lip01_for_shape = lip01

    
    # ------------------
    # Lip color (Apply to Strict Mask)
    # ------------------
    color_params = params.get("COLOR", {})

    img_colored = apply_lip_color(
        img,
        lip01, # Strict Mask
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
    # ROI Extraction
    # ------------------
    # ROI must cover the EXPANDED mask
    ys, xs = np.where(lip01_dilated > 0)
    if len(ys) == 0:
         # Fallback if mask is empty
        ys, xs = np.where(lip01 > 0)
        
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    pad = 10
    y0 = max(0, y0 - pad)
    y1 = min(H - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(W - 1, x1 + pad)

    img_colored_f = img_colored.astype(np.float32) / 255.0
    roi = img_colored_f[y0:y1+1, x0:x1+1]
    
    # Strict ROIs
    lip01_roi = lip01[y0:y1+1, x0:x1+1]
    
    # Expanded ROIs (For Shape & Clear Coat)
    lip01_shape_roi = lip01_for_shape[y0:y1+1, x0:x1+1]
    upper_shape_roi = upper_mask_for_shape[y0:y1+1, x0:x1+1]
    lower_shape_roi = lower_mask_for_shape[y0:y1+1, x0:x1+1]


    # ------------------
    # Fake normal (Calculated on Expanded Mask)
    # ------------------
    # We need normal vector even outside the original lip
    N_dict = build_layered_fake_normal(
        roi,
        lip01_shape_roi, # Use Expanded Mask
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
    # Specular (Apply to Strict Mask)
    # ------------------
    # Specular should stay inside the original lip boundary
    spec = compute_specular(
        N,
        lip01_roi, # Use Strict Mask
        params["LIGHT_DIR"],
        strength=params["SPEC_STRENGTH"],
        shininess=params["SHININESS"]
    )

    # [NEW] Colored Specular Logic
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
    # LCS (Coordinate System) - On Expanded Mask
    # ------------------
    x_hat, y_hat, r = compute_lip_coordinate_system(
        upper_shape_roi, # Use Expanded Mask
        lower_shape_roi, # Use Expanded Mask
        params["MIDLINE_BIAS"]
    )

    # ------------------
    # Clearcoat Shape (HWF) - On Expanded Mask
    # ------------------
    HWF = build_clearcoat_weight(
        x_hat, y_hat, r,
        (upper_shape_roi > 0).astype(np.float32),
        (lower_shape_roi > 0).astype(np.float32),
        lip01_shape_roi,
        lower_center=params["LOWER_CENTER"],
        upper_center=params["UPPER_CENTER"],
        lower_sigma_y=params.get("LOWER_WIDTH", 0.35),
        upper_sigma_y=params.get("UPPER_WIDTH", 0.25),
        cupid_boost=params.get("CUPID_BOOST", 0.0)
    )

    # ------------------
    # Clearcoat Apply
    # ------------------
    influence = params.get("CLEAR_NORMAL_INFLUENCE", 1.0)
    
    N_low_flat = N_low * influence + np.array([0, 0, 1]) * (1.0 - influence)
    N_low_flat /= (np.linalg.norm(N_low_flat, axis=2, keepdims=True) + 1e-6)
    
    N_clear = N_low_flat + params["ALPHA"] * N_high
    N_clear /= (np.linalg.norm(N_clear, axis=2, keepdims=True) + 1e-6)

    light_x = params.get("CLEAR_LIGHT_X", 0.0)
    L_vec = np.array([light_x, 0.0, 1.0], dtype=np.float32)
    L_vec /= np.linalg.norm(L_vec)
    
    clear_scalar = N_clear[..., 0] * L_vec[0] + N_clear[..., 2] * L_vec[2]

    # [NEW] Clear Coat Feather (Blur the Mask)
    clear_feather = params.get("CLEAR_FEATHER", 0.0)
    if clear_feather > 0:
        HWF = cv2.GaussianBlur(HWF, (0, 0), clear_feather)
        
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
    face = img_colored.astype(np.float32) / 255.0
    
    # Delta applied using EXPANDED mask logic?
    # Actually, we want to apply the clearcoat even outside the lip.
    # But Specular is only inside.
    # The 'out_roi' contains both Specular (Strict) and Clearcoat (Expanded).
    # So we should use 'lip01_shape_roi' (Expanded) for blending delta.
    
    mask_roi = (lip01_shape_roi > 0)[..., None]
    
    delta = (out_roi - roi) * mask_roi
    face[y0:y1+1, x0:x1+1] += delta
    face = np.clip(face, 0, 1)

    # Calculate final ROI for white spec comparison
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
