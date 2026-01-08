import cv2
import numpy as np
import os

def apply_lip_color_mock(
    img_bgr,
    mask,
    r,
    g,
    b,
    opacity=0.8
):
    # This is an exact copy of modules/color.py logic
    if mask.max() > 1:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)
    mask_f = mask_f[..., None]

    target_rgb = np.array([[[b, g, r]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2HSV)[0, 0]
    
    target_h = float(target_hsv[0])
    target_s = float(target_hsv[1])

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[..., 0] = target_h
    hsv_img[..., 1] = target_s

    colored = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out = (
        img_bgr.astype(np.float32) * (1.0 - mask_f * opacity)
        + colored.astype(np.float32) * (mask_f * opacity)
    )
    return np.clip(out, 0, 255).astype(np.uint8)

# Create a dummy image (Red lips on gray face)
img = np.zeros((100, 100, 3), dtype=np.uint8) + 128
mask = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(mask, (50, 50), 20, 255, -1)

# Original lip color: Red (for testing interacton, though it should be replaced)
# img[mask > 0] = [0, 0, 255] # BGR Red

# Apply Blue (#0000FF -> R=0, G=0, B=255)
res = apply_lip_color_mock(img, mask, 0, 0, 255, opacity=1.0)

# Check the color of the transformed pixels
sample_color = res[50, 50]
print(f"Result color at center: {sample_color} (BGR)")
if sample_color[0] > sample_color[2]:
    print("Looks Blueish")
elif sample_color[2] > sample_color[0]:
    print("Looks Redish/Orangeish")
else:
    print("Looks Grayish/Unknown")

# What if we swapped r and b in target_rgb?
target_rgb_swapped = np.array([[[0, 0, 255]]], dtype=np.uint8) # Pass r=0, g=0, b=255 as BGR?
target_hsv_swapped = cv2.cvtColor(target_rgb_swapped, cv2.COLOR_BGR2HSV)[0, 0]
print(f"If swapped: Hue={target_hsv_swapped[0]}")
