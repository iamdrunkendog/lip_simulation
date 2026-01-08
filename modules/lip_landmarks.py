# ==========================================
# LIP SELECTION MODULE (MIGRATED TO TASKS API)
# - MediaPipe FaceMesh (Tasks API)
# - Upper / Lower hard polygon masks
# ==========================================

import cv2
import numpy as np
import mediapipe as mp
import os

# Import Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

UPPER_LIP_IDX = [
    61,185,40,39,37,0,267,269,270,409,
    291,308,415,310,311,312,13,82,81,80,191,78
]

LOWER_LIP_IDX = [
    78,95,88,178,87,14,317,402,318,324,
    308,291,375,321,405,314,17,84,181,91,146,61
]

def _poly_from_landmarks(lms, idx_list, W, H):
    pts = []
    for i in idx_list:
        # Task API landmarks have x, y, z attributes
        x = int(np.clip(lms[i].x * W, 0, W - 1))
        y = int(np.clip(lms[i].y * H, 0, H - 1))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

def _fill_poly_mask(shape_hw, poly_pts):
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly_pts], 255)
    return m

def get_lip_masks(img_bgr):
    """
    return:
      upper_mask (uint8 0/255)
      lower_mask (uint8 0/255)
      lip_mask   (union)
      upper_poly, lower_poly
    """
    H, W = img_bgr.shape[:2]

    # Path to the model file
    # Assuming it's in the same directory as this script or modules/
    # We downloaded it to modules/face_landmarker.task
    model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    # Convert BGR to RGB and create mp.Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        raise RuntimeError("No face landmarks detected.")

    # face_landmarks is a list of lists (one per face)
    lms = detection_result.face_landmarks[0]

    upper_poly = _poly_from_landmarks(lms, UPPER_LIP_IDX, W, H)
    lower_poly = _poly_from_landmarks(lms, LOWER_LIP_IDX, W, H)

    upper_mask = _fill_poly_mask((H, W), upper_poly)
    lower_mask = _fill_poly_mask((H, W), lower_poly)
    lip_mask   = cv2.max(upper_mask, lower_mask)

    return upper_mask, lower_mask, lip_mask, upper_poly, lower_poly
