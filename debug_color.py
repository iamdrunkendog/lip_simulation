import cv2
import numpy as np

def test_color(r, g, b):
    # This is what modules/color.py does
    target_rgb = np.array([[[b, g, r]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2HSV)[0, 0]
    
    h = target_hsv[0]
    s = target_hsv[1]
    v = target_hsv[2]
    
    print(f"Input RGB: ({r}, {g}, {b})")
    print(f"Target BGR (sent to cvtColor): {target_rgb[0,0]}")
    print(f"Resulting HSV: H={h}, S={s}, V={v}")
    
    # Convert back to BGR
    hsv_back = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr_back = cv2.cvtColor(hsv_back, cv2.COLOR_HSV2BGR)[0, 0]
    print(f"Back to BGR: {bgr_back}")

print("Testing Blue #0000FF (0, 0, 255)")
test_color(0, 0, 255)

print("\nTesting Red #FF0000 (255, 0, 0)")
test_color(255, 0, 0)
