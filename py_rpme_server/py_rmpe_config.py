#!/usr/bin/env python

import numpy as np

class RmpeGlobalConfig:

    width = 368
    height = 368

    mask_stride = 8


    parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Leye", "Reye", "Lear", "Rear"]
    num_parts = len(parts)
    parts_dict = dict(zip(parts, range(num_parts)))
    print(parts_dict)
    parts += ["background"]
    num_parts_with_background = len(parts)

    # this numbers probably copied from matlab they are 1.. based not 0.. based
    limbs_conn = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]

    # correct them to good indexes
    limbs_conn = [(fr-1,to-1) for (fr,to) in limbs_conn]

    layer_idx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
              [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
              [55,56], [37,38], [45,46]]

    paf_layers = 2*len(limbs_conn)
    heat_layers = num_parts + 1
    num_layers = paf_layers + heat_layers
    paf_start = 0
    heat_start = paf_layers

    data_shape = (3, height, width)     # 3, 368, 368
    mask_shape = (height//mask_stride, height//mask_stride)  # 46, 46
    parts_shape = (num_layers, height//mask_stride, height//mask_stride)  # 1, 46, 46

    @staticmethod
    def check_layer_dictionary():

        dct = RpmeGlobalConfig.parts[:]
        dct = dct + [None]*(RpmeGlobalConfig.num_layers-len(dct))

        for (i,(fr,to)) in enumerate(RpmeGlobalConfig.limbs_conn):
            name = "%s->%s" % (RpmeGlobalConfig.parts[fr], RpmeGlobalConfig.parts[to])
            x = RpmeGlobalConfig.layer_idx[i][0]
            y = RpmeGlobalConfig.layer_idx[i][1]

            assert dct[x] is None
            dct[x] = name + ":x"
            assert dct[y] is None
            dct[y] = name + ":y"

        print(dct)


class RmpeCocoConfig:


    parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
     'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank',
     'Rank']

    num_parts = len(parts)

    # for COCO neck is calculated like mean of 2 shoulders.
    parts_dict = dict(zip(parts, range(num_parts)))

    @staticmethod
    def convert(joints):

        result = np.zeros((joints.shape[0], RmpeGlobalConfig.num_parts, 3), dtype=np.float)
        result[:,:,2]=2.  # 2 - abstent, 1 visible, 0 - invisible

        for p in RmpeCocoConfig.parts:
            coco_id = RmpeCocoConfig.parts_dict[p]
            global_id = RmpeGlobalConfig.parts_dict[p]
            assert global_id!=1, "neck shouldn't be known yet"
            result[:,global_id,:]=joints[:,coco_id,:]

        neckG = RmpeGlobalConfig.parts_dict['neck']
        RshoC = RmpeCocoConfig.parts_dict['Rsho']
        LshoC = RmpeCocoConfig.parts_dict['Lsho']


        # no neck in coco database, we calculate it as averahe of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2]<2)
        result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                    joints[both_shoulders_known, LshoC, 0:2]) / 2
        result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                 joints[both_shoulders_known, LshoC, 2])

        return result

class RpmeMPIIConfig:

    parts = ["HeadTop", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
             "RAnkle", "LHip", "LKnee", "LAnkle"]

    numparts = len(parts)

    #14 - Chest is calculated like "human center location provided by the annotated data"


    @staticmethod
    def convert(joints):
        raise "Not implemented"









# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7


if __name__ == "__main__":
    RpmeGlobalConfig.check_layer_dictionary()

