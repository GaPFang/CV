import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    ### TODO ###

    f = open(args.setting_path, 'r')
    f.readline()
    settings = f.readlines()
    f.close()

    RGB = []
    for i in range(len(settings) - 1):
        RGB.append(settings[i].split(','))
    sigma_s, sigma_r = int(settings[-1].split(',')[1]), float(settings[-1].split(',')[3])

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = []
    img_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for i in range(len(settings) - 1):
        img_gray.append(float(RGB[i][0])*img_rgb[:,:,0] + float(RGB[i][1])*img_rgb[:,:,1] + float(RGB[i][2])*img_rgb[:,:,2])
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = []
    costs = []
    for i in range(len(settings)):
        jbf_out.append(JBF.joint_bilateral_filter(img_rgb, img_gray[i]).astype(np.uint8))
        costs.append(np.abs(jbf_out[i].astype('int32')-bf_out.astype('int32')).sum())
        print('Cost: ', costs[i])
        plot = cv2.cvtColor(jbf_out[i], cv2.COLOR_RGB2BGR)
        # cv2.imwrite('jbf_' + str(args.image_path.split('/')[-1].split('.')[0]) + '_' + str(i) + '.png', plot)
    # min = np.min(costs)
    # plot = cv2.cvtColor(jbf_out[costs.index(min)], cv2.COLOR_RGB2BGR)
    # cv2.imwrite('jbf_' + str(args.image_path.split('/')[-1].split('.')[0]) + '_min.png', plot)
    # cv2.imwrite('gray_' + str(args.image_path.split('/')[-1].split('.')[0]) + '_min.png', img_gray[costs.index(min)])
    # max = np.max(costs)
    # plot = cv2.cvtColor(jbf_out[costs.index(max)], cv2.COLOR_RGB2BGR)
    # cv2.imwrite('jbf_' + str(args.image_path.split('/')[-1].split('.')[0]) + '_max.png', plot)
    # cv2.imwrite('gray_' + str(args.image_path.split('/')[-1].split('.')[0]) + '_max.png', img_gray[costs.index(max)])

if __name__ == '__main__':
    main()