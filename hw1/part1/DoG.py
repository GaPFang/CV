import numpy as np
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(threshold=500)

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        image_tmp = image
        for octave in range(self.num_octaves):
            for i in range(self.num_guassian_images_per_octave):
                if i == 0:
                    if octave != 0:
                        image_tmp = cv2.resize(gaussian_images[-1], (image_tmp.shape[1]//2, image_tmp.shape[0]//2), interpolation = cv2.INTER_NEAREST)
                    gaussian_images.append(image_tmp)
                else:
                    gaussian_images.append(cv2.GaussianBlur(image_tmp, (0, 0), self.sigma**i))

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        DoG_images = []
        for i in range(len(gaussian_images)-1):
            if i % (self.num_guassian_images_per_octave) == self.num_DoG_images_per_octave:
                continue
            DoG_images.append(cv2.subtract(gaussian_images[i], gaussian_images[i+1]))
            # print(DoG_images[-1])
            plt.imshow(DoG_images[-1], cmap='gray')
            plt.savefig('DoG' + str(int(i/self.num_guassian_images_per_octave)+1) + '-' + str(i%self.num_guassian_images_per_octave+1) + '.png')

        # Step 3: Thresholding the value and Find local extremum in 3D (local maximum and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(len(DoG_images)):
            if i%self.num_DoG_images_per_octave == 0 or i%self.num_DoG_images_per_octave == self.num_DoG_images_per_octave - 1:
                continue
            for j in range(1, DoG_images[i].shape[0]-1):
                for k in range(1, DoG_images[i].shape[1]-1):
                    if self.threshold >= abs(DoG_images[i][j,k]):
                        continue
                    max = np.max([DoG_images[i+1][j-1:j+2, k-1:k+2], DoG_images[i][j-1:j+2, k-1:k+2], DoG_images[i-1][j-1:j+2, k-1:k+2]])
                    min = np.min([DoG_images[i+1][j-1:j+2, k-1:k+2], DoG_images[i][j-1:j+2, k-1:k+2], DoG_images[i-1][j-1:j+2, k-1:k+2]])

                    if DoG_images[i][j,k] == max or DoG_images[i][j,k] == min:
                        keypoints.append([j * int(i/self.num_DoG_images_per_octave + 1), k * int(i/self.num_DoG_images_per_octave + 1)])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
