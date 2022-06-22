import os
import sys

import cv2 as cv
import numpy as np
import kitti_dataHandler as kh
from matplotlib import pyplot as plt


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    # Test
    disp_dir = 'data/test/disparity'
    output_dir = 'data/test/est_depth'
    calib_dir = 'data/test/calib'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        disp_path = disp_dir + '/' + sample_name + '.png'
        disp_values = np.array(cv.imread(disp_path, cv.IMREAD_GRAYSCALE).T)    # [x, y] order

        # Read calibration info
        calib_file_path = calib_dir + '/' + sample_name + '.txt'
        frame_calib = kh.read_frame_calib(calib_file_path)
        stereo_calib = kh.get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate depth (z = f*B/disp)
        depth_values = np.empty([disp_values.shape[0], disp_values.shape[1]])
        for i in range(disp_values.shape[0]):
            for j in range(disp_values.shape[1]):
                if disp_values[i, j] == 0:
                    depth_values[i, j] = 0
                else:
                    depth_values[i, j] = (stereo_calib.f * stereo_calib.baseline) / disp_values[i, j]

        # Compare with gt
        # gt_depth_path = gt_depth_dir + '/' + sample_name + '.png'
        # gt_depth = np.array(cv.imread(gt_depth_path, cv.IMREAD_GRAYSCALE).T)  # [x, y] order

        # Discard pixels past 80m
        counter = 0
        for i in range(depth_values.shape[0]):
            for j in range(depth_values.shape[1]):
                if depth_values[i, j] != 0:
                    if depth_values[i, j] > 80 or depth_values[i, j] < 0.1:
                        depth_values[i, j] = 0
                        counter = counter + 1

        print('Number of values over 80 [m] or less 0.1 [m]: ', counter)

        # Save depth map
        est_depth_path = output_dir + '/' + sample_name + '.png'
        cv.imwrite(est_depth_path, depth_values.T)


if __name__ == '__main__':
    main()