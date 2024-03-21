import numpy as np
import cv2
import matplotlib.pyplot as plt

# time: 1.84 sec

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32) / 255
        
        ### TODO ###
        Gs = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                Gs[i, j] = np.exp(np.divide(np.square(i-self.pad_w) + np.square(j-self.pad_w),-2*np.square(self.sigma_s)))

        # exp_term = np.exp(-np.square(np.arange(-self.pad_w, self.pad_w+1)) / (2*self.sigma_s**2))
        # Gs = np.outer(exp_term, exp_term)

        output = np.zeros(img.shape)
        for i in range(self.pad_w, padded_img.shape[0]-self.pad_w):
            for j in range(self.pad_w, padded_img.shape[1]-self.pad_w):
                Tp = padded_guidance[i,j]
                Tq = padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                if len(guidance.shape)==3:
                    Gr = np.exp(np.divide(np.sum(np.square(Tp-Tq), axis=2), -2*np.square(self.sigma_r)))
                else:
                    Gr = np.exp(np.divide(np.square(Tp-Tq), -2*np.square(self.sigma_r)))
                G = np.multiply(Gs, Gr)
                weight = np.sum(G)
                for k in range(3):
                    output[i-self.pad_w, j-self.pad_w, k] = np.sum(np.multiply(G, padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, k])) / weight
        
        # plt.imshow(output)
        plot = cv2.cvtColor(np.clip(output, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if len(guidance.shape) == 2:
            # plt.savefig('output_gray_guidance.png')
            cv2.imwrite('output_gray_guidance.png', plot)
        else:
            # plt.savefig('output_rgb_guidance.png')
            cv2.imwrite('output_rgb_guidance.png', plot)
        return np.clip(output, 0, 255).astype(np.uint8)
