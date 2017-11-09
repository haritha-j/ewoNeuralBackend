#!/usr/bin/env python

import numpy as np
from math import cos, sin, pi
import cv2
import random

from py_rmpe_config import RmpeGlobalConfig

class TransformationParams:

    stride = 8
    crop_size_x = RmpeGlobalConfig.width # = 368
    crop_size_y = RmpeGlobalConfig.height # = 368
    target_dist = 0.6;
    scale_prob = 1;
    scale_min = 0.5;
    scale_max = 1.1;
    max_rotate_degree = 40.
    center_perterb_max = 40.
    #do_clahe = False; - not yet used
    #mirror = True; - not yet used
    flip_prob = 0.5
    sigma = 7.


class AugmentSelection:

    def __init__(self, flip=False, degree = 0., crop = (0,0), scale = 1.):
        self.flip = flip
        self.degree = degree #rotate
        self.crop = crop #shift actually
        self.scale = scale

    @staticmethod
    def random():
        flip = random.uniform(0.,1.) > TransformationParams.flip_prob
        degree = random.uniform(-1.,1.) * TransformationParams.max_rotate_degree
        scale = (TransformationParams.scale_max - TransformationParams.scale_min)*random.uniform(0.,1.)+TransformationParams.scale_min \
            if random.uniform(0.,1.)> TransformationParams.scale_prob else 1.
        x_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);
        y_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    def affine(self, center, scale_self):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        A = self.scale * cos(self.degree / 180. * pi ) * TransformationParams.target_dist / scale_self # TODO: looks like my images 2 times larger than needed. Fing why
        B = self.scale * sin(self.degree / 180. * pi ) * TransformationParams.target_dist / scale_self

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]


        # TODO: implement flip
        aff = np.array( [[ A, B, (1-A)*center_x - B * center_y + TransformationParams.crop_size_x/2-center_x],
                        [ -B, A, B*center_x + (1-A) * center_y + TransformationParams.crop_size_y/2-center_y]])

        #print("center", center_x, center_y)
        #print("scale", self.scale)
        #print("aff:", aff)

        return aff


class Transformer:

    @staticmethod
    def transform(img, mask, meta, aug=AugmentSelection.random()):

        # warp picture and mask
        M = aug.affine(meta['objpos'][0], meta['scale_provided'][0])
        #M = np.array([[0.9,0.,0.],[0.,0.5,150.]])

        # TODO: need to understad this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
        # print(img.shape)
        img = cv2.warpAffine(img, M, (TransformationParams.crop_size_y, TransformationParams.crop_size_x), borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        mask = cv2.warpAffine(mask, M, (TransformationParams.crop_size_y, TransformationParams.crop_size_x), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        #_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        #assert np.all((mask == 0) | (mask == 255)), "Interpolation of mask should be thresholded only 0 or 255"


        MA = M[(0,1),:]

        # TODO: implement flip - we need not only change coordinates but also change left and right joints

        # warp key points
        #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
        original_points = meta['joints'].copy()
        original_points[:,:,2]=1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
        meta['joints'][:,:,0:2]=converted_points

        #print(meta['joints'])

        return img, mask, meta

