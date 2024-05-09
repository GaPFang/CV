import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2 * i, :] = np.array([u[i, 0], u[i, 1], 1, 0, 0, 0, -u[i, 0] * v[i, 0], -u[i, 1] * v[i, 0], -v[i, 0]])
        A[2 * i + 1, :] = np.array([0, 0, 0, u[i, 0], u[i, 1], 1, -u[i, 0] * v[i, 1], -u[i, 1] * v[i, 1], -v[i, 1]])
        
    # TODO: 2.solve H with A
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    Ux, Uy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    U = np.vstack((Ux.flatten(), Uy.flatten(), np.ones((1, Ux.size)))).T

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H_inv, U.T)
        V = V/V[2]
        Vx = V[0].reshape((ymax - ymin, xmax - xmin))
        Vy = V[1].reshape((ymax - ymin, xmax - xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (Vx >= 0) & (Vx < w_src - 1) & (Vy >= 0) & (Vy < h_src - 1)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        Vx = Vx[mask]
        Vy = Vy[mask]
        Vx_int = Vx.astype(int)
        Vy_int = Vy.astype(int)

        dVx = (Vx - Vx_int).reshape(-1, 1)
        dVy = (Vy - Vy_int).reshape(-1, 1)

        src[Vy_int, Vx_int]

        interpolated = np.zeros((h_src, w_src, ch))
        interpolated[Vy_int, Vx_int] += (1 - dVx) * (1 - dVy) * src[Vy_int, Vx_int]
        interpolated[Vy_int, Vx_int] += dVx * (1 - dVy) * src[Vy_int, Vx_int + 1]
        interpolated[Vy_int, Vx_int] += (1 - dVx) * dVy * src[Vy_int + 1, Vx_int]
        interpolated[Vy_int, Vx_int] += dVx * dVy * src[Vy_int + 1, Vx_int + 1]
        
        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][mask] = interpolated[Vy_int, Vx_int]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U.T)
        V = (V/V[2]).astype(int)
        Vx = V[0].reshape((ymax - ymin, xmax - xmin))
        Vy = V[1].reshape((ymax - ymin, xmax - xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (Vx >= 0) & (Vx < w_dst) & (Vy >= 0) & (Vy < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        Vx = Vx[mask]
        Vy = Vy[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[Vy, Vx, :] = src[mask]

    return dst 
