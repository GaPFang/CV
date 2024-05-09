import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    good_matches_num = 100
    iterate_num = 1000
    sample_num = 10
    dist_threshold = 0.1

    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx + 1]
        im2 = imgs[idx]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:good_matches_num]
        q_pts = [match.queryIdx for match in good_matches]
        t_pts = [match.trainIdx for match in good_matches]
        src_pts = np.array([kp1[i].pt for i in q_pts])
        dst_pts = np.array([kp2[i].pt for i in t_pts])

        # TODO: 2. apply RANSAC to choose best H
        best_inliers = 0
        for _ in range(iterate_num):
            sample_idx = random.sample(range(len(good_matches)), sample_num)
            H = solve_homography(src_pts[sample_idx], dst_pts[sample_idx])
            U = np.vstack((src_pts.T, np.ones((1, src_pts.shape[0]))))
            V = np.dot(H, U)
            V /= V[2]
            V = V[:2]
            V = V.T
            dist = np.linalg.norm(dst_pts - V, axis=1)
            inliers = np.sum(dist < dist_threshold)
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H.copy()

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        dst = warping(im1, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return dst 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)