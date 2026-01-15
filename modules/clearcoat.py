import numpy as np

def build_clearcoat_weight(
    x_hat, y_hat, r, upper, lower, lip01,
    lower_center=0.55, lower_sigma_y=0.35, lower_sigma_x=0.85,
    upper_center=-0.8, upper_sigma_y=0.25, upper_sigma_x=0.35,
    edge_gamma=0.3,
    cupid_boost=0.0
):
    edge = (1 - r) ** edge_gamma

    blob = np.exp(-0.5 * ((y_hat - lower_center)/lower_sigma_y)**2)
    blob *= np.exp(-0.5 * (x_hat/lower_sigma_x)**2)
    blob *= lower

    strip = np.exp(-0.5 * ((y_hat - upper_center)/upper_sigma_y)**2)
    strip *= np.exp(-0.5 * ((x_hat + 0.35)/upper_sigma_x)**2)
    strip *= upper

    HWF = np.clip(blob + strip, 0, 1)
    
    # [NEW] Cupid Boost (Specific boost for cupid's bow area)
    # y_hat is approx -1.0 at the top edge of the upper lip
    if cupid_boost > 0:
        cupid_factor = np.exp(-0.5 * ((y_hat - (-1.0)) / 0.25)**2)
        # Apply boost primarily to the upper lip mask
        HWF += cupid_factor * cupid_boost * upper
        HWF = np.clip(HWF, 0, 1)

    HWF *= edge * lip01
    return HWF


def apply_clearcoat(Nz, HWF, strength=0.4, exponent=30):
    """
    표면 반사광(Clear Coat) 적용. 
    기존 Nz(법선 벡터의 Z축)를 활용하여 정면을 향한 부위에 광택을 부여합니다.
    """
    return (np.clip(Nz,0,1)**exponent) * HWF * strength
