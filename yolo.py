#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

from __future__ import print_function, division
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

import cv2

import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core

import matplotlib.pyplot as plt

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolov3-kitti.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/kitti_classes.txt'
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, image_r=None, image_d=None, calib=None, disparity=None, xyz=None, use_camera=False):
        start = time.time()
        kitti_format_predictions = list()

        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), image.filename))
        print('Found {} boxes'.format(len(out_boxes)))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if use_camera:
                fx_l = float(calib['left']['fx'])
                fy_l = float(calib['left']['fy'])
                cx_l = float(calib['left']['cx'])
                cy_l = float(calib['left']['cy'])
                baseline = float(calib['stereo']['baseline'])

                col_l = (left + right) // 2
                row_l = (top + bottom) // 2

                disparity_min = float("inf")

                for i in range(top, bottom + 1):
                    for j in range(left, right + 1):
                        disp = float(disparity[i, j])

                        if disp < disparity_min:
                            disparity_min = disp

                Z = (fx_l * baseline) / -disparity_min
                Y = (col_l - cx_l) * Z / fx_l
                X = (row_l - cy_l) * Z / fy_l

                X = xyz[0]
                Y = xyz[1]
                Z = xyz[2]


            else:
                X = xyz[0]
                Y = xyz[1]
                Z = xyz[2]

            kitti_format_predictions.append('{} {} {} {} {} {} {} {} {} {} {} {:.2f} {:.2f} {:.2f} {} {:.2f}'.format(
                predicted_class, -1, -1, -10.0, left, top, right, bottom, -1, -1, -1, X, Y, Z, -1, score))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = time.time()
        print(end - start)
        return image, kitti_format_predictions

    def close_session(self):
        self.sess.close()


def detect_video(yolo):

    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720
    init_params.camera_fps = 60

    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    frame = core.PyMat()

    # Create and set PyRuntimeParameters after opening the camera
    runtime_parameters = zcam.PyRuntimeParameters()
    runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:

            # A new image is available if grab() returns PySUCCESS
            zed.retrieve_image(frame, sl.PyVIEW.PyVIEW_LEFT)
            image = Image.fromarray(frame.get_data())

            image, voc_detections = detect_img(yolo, image.copy(), video=True)

            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



def detect_img(yolo, img_left, img_right=None, depth=None, calib=None, disparity=None, xyz=None, video=False):

    image = img_left
    voc_predictions = list()

    try:
        if not video:
            if not os.path.exists(img_left):
                return image, voc_predictions

            img_left = Image.open(img_left)

        if img_right is not None:
            img_right = Image.open(img_right)

        if img_right is not None:
            depth = Image.open(depth)

        if disparity is not None:
            disparity = np.load(disparity)

    except:
        print('Open Error! Try again!')
        return image, voc_predictions

    else:
        image, voc_predictions = yolo.detect_image(img_left)

        # image, voc_predictions = yolo.detect_image(img_left, image_r=img_right, image_d=depth, calib=calib, disparity=disparity, xyz=xyz)
        # image.show()

        print(voc_predictions)

        return image, voc_predictions

def capture_images_zed(model):
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720
    init_params.camera_fps = 60

    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    image = core.PyMat()
    image_r = core.PyMat()
    depth = core.PyMat()
    disparity = core.PyMat()
    xyz = core.PyMat()

    # Create and set PyRuntimeParameters after opening the camera
    runtime_parameters = zcam.PyRuntimeParameters()
    runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

    i = 0
    while i < 500:
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # A new image is available if grab() returns PySUCCESS
            zed.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT)
            zed.retrieve_image(image_r, sl.PyVIEW.PyVIEW_RIGHT)

            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)

            # Retrieve disparity
            zed.retrieve_measure(disparity, sl.PyMEASURE.PyMEASURE_DISPARITY)

            # Retrieve X, Y, Z
            zed.retrieve_measure(xyz, sl.PyMEASURE.PyMEASURE_XYZ)

            timestamp = zed.get_timestamp(
                sl.PyTIME_REFERENCE.PyTIME_REFERENCE_CURRENT)  # Get the timestamp at the time the image was captured
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                                                                                 timestamp))

            im_l = Image.fromarray(image.get_data())
            im_r = Image.fromarray(image_r.get_data())
            im_d = np.dstack((depth.get_data(), depth.get_data(), depth.get_data()))

            disparity_data = disparity.get_data()

            xyz_data = xyz.get_data()

            im_l.save('/home/sam/ownCloud/Deep Learning Models/keras-yolo3/model_data/zed/zed_{}_left.png'.format(i), 'PNG')
            im_r.save('/home/sam/ownCloud/Deep Learning Models/keras-yolo3/model_data/zed/zed_{}_right.png'.format(i), 'PNG')
            cv2.imwrite('/home/sam/ownCloud/Deep Learning Models/keras-yolo3/model_data/zed/zed_{}_depth.png'.format(i), im_d)
            np.save('/home/sam/ownCloud/Deep Learning Models/keras-yolo3/model_data/zed/zed_{}_disparity.npy'.format(i), disparity_data)
            np.save('/home/sam/ownCloud/Deep Learning Models/keras-yolo3/model_data/zed/zed_{}_xyz.npy'.format(i), xyz_data)
            i += 1

    zed.close()


def parse_zed_calib(calib_file, res='FHD'):
    calib = dict()

    with open(calib_file, 'r') as f:
        lines = f.readlines()

        idx = 0

        while idx < len(lines):
            line = lines[idx]

            if line in ['\n', '\r\n']:
                idx += 1
                continue

            if res == '2K' and res in line:
                cam = 'left' if 'LEFT' in line.split('_')[0] else 'right'
                locations = parse_calib_cam_info(lines[idx+1:idx+7])
                calib[cam] = locations
                idx += 6

            elif res == 'FHD' and res in line:
                cam = 'left' if 'LEFT' in line.split('_')[0] else 'right'
                locations = parse_calib_cam_info(lines[idx+1:idx+7])
                calib[cam] = locations
                idx += 6

            elif 'FHD' not in line and res == 'HD' and res in line:
                cam = 'left' if 'LEFT' in line.split('_')[0] else 'right'
                locations = parse_calib_cam_info(lines[idx+1:idx+7])
                calib[cam] = locations
                idx += 6

            elif res == 'VGA' and res in line:
                cam = 'left' if 'LEFT' in line.split('_')[0] else 'right'
                locations = parse_calib_cam_info(lines[idx + 1:idx + 7])
                calib[cam] = locations
                idx += 6

            elif 'STEREO' in line:
                calib['stereo'] = parse_calib_stereo_info(lines[idx+1:idx+13], res)
                idx += 13

            elif '2K' in line or 'FHD' in line or 'HD' in line or 'VGA' in line:
                idx += 6

            idx += 1

    return calib


def parse_calib_cam_info(cam_locations):

    locations = dict()
    locations['cx'] = cam_locations[0].split('=')[-1].strip()
    locations['cy'] = cam_locations[1].split('=')[-1].strip()
    locations['fx'] = cam_locations[2].split('=')[-1].strip()
    locations['fy'] = cam_locations[3].split('=')[-1].strip()
    locations['k1'] = cam_locations[4].split('=')[-1].strip()
    locations['k2'] = cam_locations[5].split('=')[-1].strip()

    return locations


def parse_calib_stereo_info(stereo_info, res):

    stereo = dict()
    idx = 0

    while idx < len(stereo_info):

        line = stereo_info[idx]

        if 'BaseLine' in line:
            stereo['baseline'] = line.split('=')[-1].strip()

        elif res in line and 'CV' in line:
            stereo['cv'] = line.split('=')[-1].strip()

        elif res in line and 'RX' in line:
            stereo['rx'] = line.split('=')[-1].strip()

        elif res in line and 'RZ' in line:
            stereo['rz'] = line.split('=')[-1].strip()

        idx += 1

    return stereo



if __name__ == '__main__':
    m = YOLO()

    calib_file = '/usr/local/zed/settings/SN17359.conf' # camera calibration file
    calib = parse_zed_calib(calib_file)


    img_l = './model_data/000000.png'
    image, _ = detect_img(m, img_l, calib=calib)

    img_l = './model_data/000001.png'
    image1, _ = detect_img(m, img_l, calib=calib)

    img_l = './model_data/000002.png'
    image2, _ = detect_img(m, img_l, calib=calib)

    img_l = './model_data/000016.png'
    image3, _ = detect_img(m, img_l, calib=calib)

    plt.imshow(image)
    plt.show()
    plt.imshow(image1)
    plt.show()
    plt.imshow(image2)
    plt.show()
    plt.imshow(image3)

    plt.show()


    # for i in range(500):
    #     img_l = './model_data/zed/zed_{}_left.png'.format(i)
        # img_r = './model_data/zed/zed_{}_right.png'.format(i)
        # depth = './model_data/zed/zed_{}_depth.png'.format(i)
        # disparity = './model_data/zed/zed_{}_disparity.npy'.format(i)
        # xyz = './model_data/zed/zed_{}_xyz.npy'.format(i)

        # result, predictions = detect_img(m, img_l, img_r, depth, calib, disparity, xyz)
        # result, predictions = detect_img(m, img_l)

        # if len(predictions) > 0:
        #     result.save()
        #     cv2.imwrite('./model_data/zed/zed_{}_result.png'.format(i), result)
    #
    # detect_video(m)

    # zed_camera(m)

    m.close_session()