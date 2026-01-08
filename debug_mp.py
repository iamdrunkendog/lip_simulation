import mediapipe as mp
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("Tasks API imported successfully")
    print(f"FaceLandmarker available: {hasattr(vision, 'FaceLandmarker')}")
except ImportError as e:
    print(f"ImportError: {e}")
