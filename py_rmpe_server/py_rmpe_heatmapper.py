#!/usr/bin/env python

import numpy as np
from math import sqrt, isnan

from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig, TransformationParams

class Heatmapper:

    def __init__(self, sigma=TransformationParams.sigma, thre=TransformationParams.paf_thre):

        self.double_sigma2 = 2 * sigma * sigma
        self.thre = thre

        # cached common parameters which same for all iterations and all pictures

        stride = RmpeGlobalConfig.stride
        width = RmpeGlobalConfig.width//stride
        height = RmpeGlobalConfig.height//stride

        # this is coordinates of centers of bigger grid
        self.grid_x = np.arange(width)*stride + stride/2-0.5
        self.grid_y = np.arange(height)*stride + stride/2-0.5

        self.Y, self.X = np.mgrid[0:RmpeGlobalConfig.height:stride,0:RmpeGlobalConfig.width:stride]
        self.X = self.X + stride / 2 - 0.5
        self.Y = self.Y + stride / 2 - 0.5

    def create_heatmaps(self, joints, mask):

        heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)

        self.put_joints(heatmaps, joints)
        sl = slice(RmpeGlobalConfig.heat_start, RmpeGlobalConfig.heat_start + RmpeGlobalConfig.heat_layers)
        heatmaps[RmpeGlobalConfig.bkg_start] = 1. - np.amax(heatmaps[sl,:,:], axis=0)

        self.put_limbs(heatmaps, joints)

        return heatmaps * mask

    def put_gaussian_maps(self, heatmaps, layer, joints):

        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by

        for i in range(joints.shape[0]):

            exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)

            exp = np.outer(exp_y, exp_x)

            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[RmpeGlobalConfig.heat_start + layer, :, :] = np.maximum(heatmaps[RmpeGlobalConfig.heat_start + layer, :, :], exp)

    def put_joints(self, heatmaps, joints):

        for i in range(RmpeGlobalConfig.num_parts):
            visible = joints[:,i,2] < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])


    def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):

        count = np.zeros(heatmaps.shape[1:], dtype=np.int)

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2-x1
            dy = y2-y1
            dnorm = sqrt(dx*dx + dy*dy)

            if dnorm==0:  # we get nan here sometimes, it's kills NN
                # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely
                print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            # TODO: check PAF off screen.
            # looks like slices working properly
            # > foo=np.array([1,2,3,4,5,6,7,8,9,10])
            # > foo[slice(-10,5)]
            # array([1, 2, 3, 4, 5])

            min_sx = int((min_sx - self.thre) / RmpeGlobalConfig.stride)
            min_sy = int((min_sy - self.thre) / RmpeGlobalConfig.stride)
            max_sx = int((max_sx + self.thre) / RmpeGlobalConfig.stride)
            max_sy = int((max_sy + self.thre) / RmpeGlobalConfig.stride)

            slice_x = slice(min_sx, max_sx + 1)
            slice_y = slice(min_sy, max_sy + 1)

            dist = distances(self.X[slice_y,slice_x], self.Y[slice_y,slice_x], x1, y1, x2, y2)
            dist = dist < RmpeGlobalConfig.stride

            heatmaps[layerX, slice_y, slice_x] += dist * dx
            heatmaps[layerY, slice_y, slice_x] += dist * dy
            count[slice_y, slice_x] += 1

        heatmaps[layerX, :, :][count > 0] /= count[count > 0]
        heatmaps[layerY, :, :][count > 0] /= count[count > 0]

    def put_limbs(self, heatmaps, joints):

        for (i,(fr,to)) in enumerate(RmpeGlobalConfig.limbs_conn):


            visible_from = joints[:,fr,2] < 2
            visible_to = joints[:,to, 2] < 2
            visible = visible_from & visible_to

            layerX, layerY = (RmpeGlobalConfig.paf_start + i*2, RmpeGlobalConfig.paf_start + i*2 + 1)
            self.put_vector_maps(heatmaps, layerX, layerY, joints[visible, fr, 0:2], joints[visible, to, 0:2])



#parallel calculation distance from any number of points of arbitrary shape(X, Y), to line defined by segment (x1,y1) -> (x2, y2)

def distances(X, Y, x1, y1, x2, y2):

    # classic formula is:
    # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)

    xD = (x2-x1)
    yD = (y2-y1)
    norm2 = sqrt(xD**2 + yD**2)
    dist = xD*(y1-Y)-(x1-X)*yD
    dist /= norm2

    return np.abs(dist)

def test():

    hm = Heatmapper()
    d = distances(hm.X, hm.Y, 100, 100, 50, 150)
    print(d < 8.)

if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
    test()

