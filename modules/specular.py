import numpy as np

def screen(base, add):
    return 1.0 - (1.0 - base) * (1.0 - add)


def compute_specular(N, lip01, light_dir, strength=0.18, shininess=60):
    Nx, Ny, Nz = N[...,0], N[...,1], N[...,2]
    Lx, Ly, Lz = light_dir

    ndotl = np.clip(Nx*Lx + Ny*Ly + Nz*Lz, 0, 1)
    spec = (ndotl ** shininess) * strength
    return spec * lip01
