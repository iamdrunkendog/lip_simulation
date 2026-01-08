import cv2
import numpy as np


def make_soft_mask(
    binary_mask,
    edge_width=6,
    smoothstep=True
):
    """
    binary_mask : uint8 (0 or 255)
    edge_width  : 경계 부드러움 픽셀 거리
    smoothstep  : perceptual falloff 적용 여부
    return      : float32 mask (0~1)
    """

    if binary_mask.dtype != np.uint8:
        raise ValueError("binary_mask must be uint8 (0 or 255)")

    bin01 = (binary_mask > 0).astype(np.uint8)

    # distance inside / outside
    dist_in = cv2.distanceTransform(bin01, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - bin01, cv2.DIST_L2, 5)

    signed_dist = dist_in - dist_out

    # normalize around boundary
    # Old Inward: np.clip(signed_dist / edge_width, 0.0, 1.0)
    # Old Outward: np.clip(1.0 + (signed_dist / edge_width), 0.0, 1.0)
    # New Centered: 0.5 at boundary, fades in and out
    soft = np.clip(0.5 + (signed_dist / edge_width), 0.0, 1.0)

    if smoothstep:
        # smoothstep: 3t^2 - 2t^3
        soft = soft * soft * (3.0 - 2.0 * soft)

    return soft.astype(np.float32)


def combine_upper_lower(
    upper_mask,
    lower_mask,
    edge_width_upper=6,
    edge_width_lower=8
):
    """
    상/하 입술을 서로 다른 falloff로 결합
    return: lip01 (float32 0~1)
    """
    upper_soft = make_soft_mask(
        upper_mask,
        edge_width=edge_width_upper
    )
    lower_soft = make_soft_mask(
        lower_mask,
        edge_width=edge_width_lower
    )

    lip01 = np.clip(upper_soft + lower_soft, 0.0, 1.0)
    return lip01
