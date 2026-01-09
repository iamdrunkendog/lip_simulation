import numpy as np

def screen(base, add):
    return 1.0 - (1.0 - base) * (1.0 - add)


def add_blend(base, add):
    return np.clip(base + add, 0, 1)


def color_dodge(base, add):
    # base / (1 - add)
    # add가 1에 가까울수록 매우 밝아지며 원본 색상이 강하게 배어나옴
    return np.where(add < 1.0, np.clip(base / (1.0 - add + 1e-6), 0, 1), 1.0)


def overlay(base, add):
    return np.where(base < 0.5, 2.0 * base * add, 1.0 - 2.0 * (1.0 - base) * (1.0 - add))


def blend_specular(base, spec, mode="screen"):
    """
    spec레이어를 base 이미지에 특정 모드로 합성합니다.
    spec은 (H, W) 또는 (H, W, 1) 형태여야 하며, 내부적으로 채널 확장 처리됩니다.
    """
    if spec.ndim == 2:
        spec = spec[..., None]
        
    mode = mode.lower()
    if mode == "add":
        return add_blend(base, spec)
    elif mode == "color_dodge":
        return color_dodge(base, spec)
    elif mode == "overlay":
        return overlay(base, spec)
    elif mode == "normal":
        # 단순 덮어씌우기 (비권장)
        return np.clip(base * (1.0 - spec) + spec, 0, 1)
    else: # default is screen
        return screen(base, spec)


def compute_specular(N, lip01, light_dir, strength=0.18, shininess=60, mask=None):
    """
    Blinn-Phong 기반 Specular 계산.
    strength가 1.0을 넘어가면 Screen 블렌딩 시 하아라이트 중심부가 순백색(White)으로 수렴합니다.
    """
    Nx, Ny, Nz = N[...,0], N[...,1], N[...,2]
    Lx, Ly, Lz = light_dir

    ndotl = np.clip(Nx*Lx + Ny*Ly + Nz*Lz, 0, 1)
    spec = (ndotl ** shininess) * strength
    
    if mask is not None:
        spec *= mask
        
    return spec * lip01
