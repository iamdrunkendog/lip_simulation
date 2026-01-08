import cv2
import numpy as np

def debug_apply_lip_color(r, g, b):
    # This is an exact copy of modules/color.py logic
    target_rgb = np.array([[[b, g, r]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2HSV)[0, 0]
    
    th, ts, tv = target_hsv
    print(f"Input RGB: ({r}, {g}, {b})")
    print(f"Target BGR array: {target_rgb[0,0]}")
    print(f"Target HSV: H={th}, S={ts}, V={tv}")

    # Dummy source image (pure gray)
    img_bgr = np.zeros((1, 1, 3), dtype=np.uint8) + 128
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    hsv_img[..., 0] = th
    hsv_img[..., 1] = ts
    # hsv_img[..., 2] stays 128
    
    colored = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
    print(f"Colored (BGR): {colored[0,0]}")

print("--- Testing Blue #0000FF ---")
debug_apply_lip_color(0, 0, 255)

print("\n--- Testing Red #FF0000 ---")
debug_apply_lip_color(255, 0, 0)
