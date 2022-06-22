import os
import sys

import cv2
import numpy as np
import kitti_dataHandler as kh
import skimage.segmentation as seg
from matplotlib import pyplot as plt


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    depth_dir = 'data/test/est_depth'
    boxes_dir = 'data/test/output_box'
    output_dir = 'data/test/est_segmentation'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
        # Read depth map
        depth_path = depth_dir + '/' + sample_name + '.png'
        gt_depth = np.array(cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).T)  # [x, y] order

        # Read 2d bbox
        boxes_path = boxes_dir + '/' + sample_name + '.npy'
        boxes = np.load(boxes_path)

        # Load Image
        image_path = 'data/test/left/' + sample_name + '.png'
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)

        # Create a mask
        mask = np.ones([img.shape[0], img.shape[1]]) * 255

        # Initialize output figure (No axes, no blank space considered for dimensions)
        fig = plt.figure(figsize=(15.78, 4.62))
        ax = plt.axes()
        ax.set_axis_off()

        # Obtain bboxes information
        for i in range(len(boxes)):
            (x_min, y_min) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            x_max = x_min + w
            y_max = y_min + h

            # Consider boxes that exceed limits of the depth maps
            if x_max > gt_depth.shape[0]:
                x_max = gt_depth.shape[0]

            if y_max > gt_depth.shape[1]:
                y_max = gt_depth.shape[1]

            # Obtain arrays of perimeter points of bboxes by side
            points_up = []
            for j in range(x_min, x_max):
                points_up.append([j, y_min])

            points_up = np.array(points_up)

            points_d = []
            for j in range(x_min, x_max):
                points_d.append([j, y_max])

            points_d = np.array(points_d)

            points_le = []
            for j in range(y_min + 1, y_max - 1):
                points_le.append([x_min, j])

            points_le = np.array(points_le)

            points_r = []
            for j in range(y_min + 1, y_max - 1):
                points_r.append([x_max, j])

            points_r = np.array(points_r)

            # Concatenate arrays
            points = np.concatenate((points_up, points_r, points_d[::-1], points_le[::-1]), axis=0)

            # Obtain average depth for segmentation parameters
            bbox_depth = []
            for j in range(x_min, x_max):
                for k in range(y_min, y_max):
                    # Include only values != zero for depth mean
                    if gt_depth[j, k] != 0:
                        bbox_depth.append(gt_depth[j, k])

            bbox_depth = np.array(bbox_depth)
            if len(bbox_depth) == 0:
                average_depth = 50
            else:
                average_depth = np.around(np.average(bbox_depth), 0)

            print(average_depth)

            # Set alpha and beta parameters based on average depth value
            if average_depth <= 10:
                a = 1
                b = 1

            if 10 < average_depth <= 20:
                a = 0.05
                b = 25

            if average_depth > 20:
                a = 0.05
                b = 15

            snake = seg.active_contour(img, points, alpha=a, beta=b, gamma=0.01)
            snake = np.array(snake)
            snake = np.around(np.vstack((snake, snake[0])), 0)
            polygons = plt.fill(snake[:, 0], snake[:, 1], color='k')

        plt.imshow(mask, cmap='gray', vmin=0, vmax=255, aspect='auto')
        plt.savefig('test.png', bbox_inches='tight')
        img2 = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
        ar = np.array(img2)
        ar[ar < 255] = 0
        output_path = output_dir + '/' + sample_name + '.png'
        cv2.imwrite(output_path, ar)

if __name__ == '__main__':
    main()