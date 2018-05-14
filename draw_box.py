
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont


def draw_3d_box(image_file, label_file, calib_file):
    image = cv2.imread(image_file)
    cam_to_img = None

    # read calibration data
    with open(calib_file, 'r') as f:
        for line in open(calib_file):
            if 'P2:' in line:
                cam_to_img = line.strip().split(' ')
                cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
                cam_to_img = np.reshape(cam_to_img, (3, 4))

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')

            # Draw 3D Bounding Box
            dims = np.asarray([float(number) for number in line[8:11]])
            center = np.asarray([float(number) for number in line[11:14]])

            if np.abs(float(line[3])) < 0.01:
                continue
            print(line[3], center)

            rot_y = float(line[3]) + np.arctan(center[0] / center[2])  # float(line[14])

            box_3d = []

            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [0, 1]:
                        point = np.copy(center)
                        point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + (j * i) * dims[
                            2] / 2 * np.cos(-rot_y)
                        point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + (j * i) * dims[
                            2] / 2 * np.sin(-rot_y)
                        point[1] = center[1] - k * dims[0]

                        point = np.append(point, 1)
                        point = np.dot(cam_to_img, point)
                        point = point[:2] / point[2]
                        point = point.astype(np.int16)
                        box_3d.append(point)

            for i in range(4):
                point_1_ = box_3d[2 * i]
                point_2_ = box_3d[2 * i + 1]
                cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

            for i in range(8):
                point_1_ = box_3d[i]
                point_2_ = box_3d[(i + 2) % 8]
                cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 3)

        plt.imshow(image)
        plt.show()


def draw_multiple_3d_box(img_dir, label_dir, calib_dir):
    images = [filename.split('.')[0] for filename in sorted(os.listdir(label_dir))]

    for image_id in images:

        image_file = os.path.join(img_dir, image_id + '.png')
        label_file = os.path.join(label_dir, image_id + '.txt')
        calib_file = os.path.join(calib_dir, image_id + '.txt')

        draw_3d_box(image_file, label_file, calib_file)



if __name__ == '__main__':
    img_dir = '/home/sam/ownCloud/Deep Learning Models/3D-Deepbox/training/image_2/'
    label_dir = '/home/sam/ownCloud/Deep Learning Models/3D-Deepbox/training/label_2/'
    calib_dir = '/home/sam/ownCloud/Deep Learning Models/3D-Deepbox/training/calib/'
    draw_multiple_3d_box(img_dir, label_dir, calib_dir)


