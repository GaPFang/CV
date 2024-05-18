import numpy as np
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, _ = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    sigma_r, sigma_s = 3.5, 10

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    imgl = cv2.copyMakeBorder(Il, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    imgr = cv2.copyMakeBorder(Ir, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    
    census_left = np.zeros((*imgl.shape, 25))
    census_right = np.zeros((*imgr.shape, 25))
    for i in range(-2, 3):
        for j in range(-2, 3):
            index = (i+1)*5+j+1
            census_left[:, :, :, index] = (imgl > np.roll(imgl, [i, j], axis=[0, 1])) * 1
            census_right[:, :, :, index] = (imgr > np.roll(imgr, [i, j], axis=[0, 1])) * 1
    
    census_left = census_left[2:-2, 2:-2, :, :]
    census_right = census_right[2:-2, 2:-2, :, :]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    cost_left = np.zeros((h, w, max_disp + 1))
    cost_right = np.zeros((h, w, max_disp + 1))
    for d in range(max_disp + 1):
        overlap_left = census_left[:,d:].astype(np.int8)
        overlap_right = census_right[:, :w-d].astype(np.int8)
        cost_both = np.sum(overlap_left != overlap_right, axis=2)
        cost_both = np.sum(cost_both, axis=2).astype(np.float32)
        cost_left_tmp = cv2.copyMakeBorder(cost_both, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        cost_left[:, :, d] = cv2.ximgproc.jointBilateralFilter(Il, cost_left_tmp, -1, sigma_r, sigma_s)
        cost_right_tmp = cv2.copyMakeBorder(cost_both, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        cost_right[:, :, d] = cv2.ximgproc.jointBilateralFilter(Ir, cost_right_tmp, -1, sigma_r, sigma_s)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disp_table_left = np.argmin(cost_left, axis=2)
    disp_table_right = np.argmin(cost_right, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # Left-right consistency check
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    x_right = X - disp_table_left
    mask1 = (x_right >= 0)
    disp_left = disp_table_left[mask1]
    disp_right = disp_table_right[Y[mask1], x_right[mask1]]
    mask2 = (disp_left == disp_right)
    disp_table = np.zeros((h, w), dtype=np.float32)
    disp_table[Y[mask1][mask2], X[mask1][mask2]] = disp_table_left[mask1][mask2]

    # Hole filling
    labels_left = np.copy(disp_table)
    labels_right = np.copy(disp_table)

    for y in range(h):
        left_start = False
        right_start = False
        for x in range(w):
            # print(labels_left[y, x], left_start, end=' ')
            if labels_left[y, x] == 0:
                if left_start:
                    labels_left[y, x] = labels_left[y, x-1]
                else:
                    labels_left[y, x] = max_disp
            else:
                left_start = True
            # print(labels_right[y, x])
            if labels_right[y, w-x-1] == 0:
                if right_start:
                    labels_right[y, w-x-1] = labels_right[y, w-x]
                else:
                    labels_right[y, w-x-1] = max_disp
            else:
                right_start = True
    labels = np.min((labels_left, labels_right), axis=0)

    # Weighted median filtering
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 10)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 8)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 5)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 5)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 5)
    labels = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels, 3)

    return labels.astype(np.uint8)