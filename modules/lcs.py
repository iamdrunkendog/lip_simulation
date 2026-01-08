import numpy as np

def compute_lip_coordinate_system(upper_mask, lower_mask, midline_bias=0.12):
    upper = (upper_mask > 0).astype(np.uint8)
    lower = (lower_mask > 0).astype(np.uint8)
    lip = (upper + lower) > 0

    h, w = upper.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    ys_u = np.where(upper)[0]
    ys_l = np.where(lower)[0]

    y_mid = (
        (1 - midline_bias) * 0.5 * (ys_u.max() + ys_l.min())
        + midline_bias * ys_u.max()
    )

    xs = np.where(lip)[1]
    cx = xs.mean()

    x_hat = (xx - cx)
    x_hat /= np.max(np.abs(x_hat[lip])) + 1e-6
    x_hat *= lip

    y_hat = yy - y_mid
    y_hat[y_hat < 0] /= (y_mid - ys_u.min() + 1e-6)
    y_hat[y_hat > 0] /= (ys_l.max() - y_mid + 1e-6)
    y_hat = np.clip(y_hat, -1, 1) * lip

    r = np.zeros_like(y_hat)
    for y in range(h):
        xs_line = np.where(lip[y])[0]
        if len(xs_line) > 1:
            hw = 0.5 * (xs_line.max() - xs_line.min())
            r[y] = abs(y - y_mid) / (hw + 1e-6)

    r = np.clip(r, 0, 1) * lip
    return x_hat, y_hat, r
