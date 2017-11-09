
import numpy as np

from py_rmpe_config import RmpeGlobalConfig
from py_rmpe_transformer import TransformationParams

class Heatmapper:

    def __init__(self, sigma=TransformationParams.sigma):

        self.double_sigma2 = 2 * sigma * sigma

        # casched common parameters which same for all iterations and all pictures

        stride = RmpeGlobalConfig.mask_stride

        # this is coordinates of centers of bigger grid
        self.grid_y = np.arange(RmpeGlobalConfig.height//stride)*stride + stride/2-0.5
        self.grid_x = np.arange(RmpeGlobalConfig.height//stride)*stride + stride/2-0.5

        print("GRID_X:", self.grid_x)
        print("GRID_Y:", self.grid_y)


    def create_heatmaps(self, joints):

        heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)
        self.put_joints(heatmaps, joints)
        return heatmaps

    def put_gaussian_maps(self, heatmaps, layer, joints):

        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by
        if layer == 0:
            print("NOSES:", joints)

        for i in range(joints.shape[0]):

            exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)

            #if layer==0:
            #    print("%d, joint: %f, EXP_X: %s" % (layer, joints[i,0], exp_x))
            #    print("%d, joint: %f, EXP_Y: %s" % (layer, joints[i,1], exp_y))

            exp = np.outer(exp_y,exp_x)

            #if layer==0:
            #    print("%d, EXP: %s" % (layer, exp))

            heatmaps[RmpeGlobalConfig.heat_start + layer, :, :] += exp


    def put_joints(self, heatmaps, joints):

        for i in range(RmpeGlobalConfig.num_parts):
            visible = joints[:,i,2] < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])


    def put_vector_maps(self, layerX, layerY, centerA, centerB):
        pass

    def put_limbs(self):
        pass