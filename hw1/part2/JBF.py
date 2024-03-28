import numpy as np
import cv2
import matplotlib.pyplot as plt

# time: 0.46 sec

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        LUT_s = np.exp(-np.square(np.arange(self.pad_w+1)) / (2*self.sigma_s**2))
        LUT_r = np.exp(-np.square(np.arange(256) / 255) / (2*self.sigma_r**2))

        w_sum = np.zeros(padded_guidance.shape[:2])
        sum = np.zeros(padded_img.shape)
        for i in range(-self.pad_w, self.pad_w+1):
            for j in range(-self.pad_w, self.pad_w+1):
                if len(guidance.shape)==3:
                    r_w = np.prod(LUT_r[np.abs(np.roll(padded_guidance, [i,j], axis=[0,1])-padded_guidance)], axis=2)
                else:
                    r_w = LUT_r[np.abs(np.roll(padded_guidance, [i,j], axis=[0,1])-padded_guidance)]
                s_w = LUT_s[np.abs(i)] * LUT_s[np.abs(j)]
                t_w = s_w * r_w
                padded_img_roll = np.roll(padded_img, [i,j], axis=[0,1])
                w_sum += t_w
                for k in range(padded_img.shape[2]):
                    sum[:,:,k] += padded_img_roll[:,:,k] * t_w
        output = (sum / w_sum[:,:,np.newaxis])[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w]
        
        return np.clip(output, 0, 255).astype(np.uint8)
