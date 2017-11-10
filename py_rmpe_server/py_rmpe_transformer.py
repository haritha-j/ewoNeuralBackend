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
    scale_prob = 1;   # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
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
            if random.uniform(0.,1.) > TransformationParams.scale_prob else 1. # TODO: see 'scale improbability' TODO above
        x_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);
        y_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    def affine(self, center, scale_self):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards


        A = self.scale * cos(self.degree / 180. * pi )
        B = self.scale * sin(self.degree / 180. * pi )

        scale_size = TransformationParams.target_dist / scale_self * self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        scale = np.array( [[ scale_size, 0, 0 ],
                           [ 0, scale_size, 0 ],
                           [  0, 0, 1. ] ])

        flip = np.array( [[ -1 if self.flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., TransformationParams.crop_size_x//2],
                                   [ 0., 1., TransformationParams.crop_size_y//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)
        print(combined)

        return combined[0:2]

class Transformer:

    @staticmethod
    def transform(img, mask, meta, aug=AugmentSelection.random()):

        # warp picture and mask
        M = aug.affine(meta['objpos'][0], meta['scale_provided'][0])

        # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
        # print(img.shape)
        img = cv2.warpAffine(img, M, (TransformationParams.crop_size_y, TransformationParams.crop_size_x), borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        mask = cv2.warpAffine(mask, M, (TransformationParams.crop_size_y, TransformationParams.crop_size_x), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        #_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        #assert np.all((mask == 0) | (mask == 255)), "Interpolation of mask should be thresholded only 0 or 255"


        # warp key points
        #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
        original_points = meta['joints'].copy()
        original_points[:,:,2]=1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
        meta['joints'][:,:,0:2]=converted_points

        # we just made image flip, i.e. right leg just became left leg, and vice versa

        if aug.flip:
            tmpLeft = meta['joints'][:, RmpeGlobalConfig.leftParts, :]
            tmpRight = meta['joints'][:, RmpeGlobalConfig.rightParts, :]
            meta['joints'][:, RmpeGlobalConfig.leftParts, :] = tmpRight
            meta['joints'][:, RmpeGlobalConfig.rightParts, :] = tmpLeft


        return img, mask, meta

