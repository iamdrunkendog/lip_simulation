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

    y_hat = np.zeros_like(yy)
    r = np.zeros_like(yy)
    
    for x in range(w):
        ys_lip = np.where(lip[:, x])[0]
        if len(ys_lip) == 0:
            continue
            
        y_min = ys_lip.min()
        y_max = ys_lip.max()
        
        # Calculate local y_mid for this column
        # Finding the boundary between upper and lower in this column
        ys_u_col = np.where(upper[:, x])[0]
        ys_l_col = np.where(lower[:, x])[0]
        
        if len(ys_u_col) > 0 and len(ys_l_col) > 0:
            y_mid_col = (1 - midline_bias) * 0.5 * (ys_u_col.max() + ys_l_col.min()) + midline_bias * ys_u_col.max()
        elif len(ys_u_col) > 0:
            y_mid_col = ys_u_col.max()
        elif len(ys_l_col) > 0:
            y_mid_col = ys_l_col.min()
        else:
            y_mid_col = (y_min + y_max) * 0.5

        # Normalize y_hat locally
        col_y = yy[:, x]
        mask_col = lip[:, x]
        
        # Upper part
        idx_u = (col_y < y_mid_col) & mask_col
        if (y_mid_col - y_min) > 1e-6:
            y_hat[idx_u, x] = (col_y[idx_u] - y_mid_col) / (y_mid_col - y_min)
            
        # Lower part
        idx_l = (col_y >= y_mid_col) & mask_col
        if (y_max - y_mid_col) > 1e-6:
            y_hat[idx_l, x] = (col_y[idx_l] - y_mid_col) / (y_max - y_mid_col)

        # r calculation (distance from center line relative to half-height)
        y_hat[mask_col, x] = np.clip(y_hat[mask_col, x], -1, 1)
        r[mask_col, x] = np.abs(y_hat[mask_col, x])

    x_hat *= lip
    y_hat *= lip
    r *= lip
    return x_hat, y_hat, r
