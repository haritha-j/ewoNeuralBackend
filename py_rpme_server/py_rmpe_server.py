#!/usr/bin/env python
import sys
import numpy as np
import h5py
import zmq
import random
import json
import cv2

from multiprocessing import Process

from py_rmpe_config import RmpeGlobalConfig, RmpeCocoConfig

from py_rmpe_transformer import Transformer
from py_rmpe_heatmapper import Heatmapper

class Server:

    # these methods all called in parent process

    def __init__(self, h5file, port, name):

        self.name = name
        self.port = port
        self.h5file = h5file

        self.process = Process(target=Server.loop, args=(self,))
        self.process.daemon = True
        self.process.start()

        self.heatmapper = None


    def join(self):

        return self.process.join(10)

    # these methods all called in child process

    def init(self):

        self.h5 = h5py.File(self.h5file, "r")
        self.datum = self.h5['datum']

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.set_hwm(1) #TODO: put 160 in production, made for debug purposes
        self.socket.bind("tcp://*:%s" % self.port)

    @staticmethod
    def loop(self):

        print("%s: Child process init... " % self.name)
        self.init()

        print("%s: Loop started... " % self.name)

        num = 0
        generation = 0

        self.heatmapper = Heatmapper()

        while True:

            keys = list(self.datum.keys())
            #TODO: disabled for test purposes
            # random.shuffle(keys)
            print("%s: generation %s, %d images " % (self.name, generation,len(keys)))

            for key in keys:

                image, mask, meta = self.read_data(key)

                print("[in] IMAGE:", image.shape, image.dtype)
                print("[in] MASK:", mask.shape, mask.dtype)
                print("[in] meta.joints:", meta['joints'].shape, meta['joints'].dtype)

                #print(meta)
                image, mask, meta, labels = self.transform_data(image, mask, meta)

                image = np.transpose(image, (2, 0, 1))
                # TODO: should be combined with warp for speed
                mask = cv2.resize(mask, RmpeGlobalConfig.mask_shape, cv2.INTER_CUBIC)
                _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
                assert np.all((mask==0) | (mask==255)), "Interpolation of mask should be thresholded only 0 or 255"
                mask = mask.astype(np.float)/255.
                headers = self.produce_headers(image, mask, labels)

                print("[out] HEADERS:",headers)
                self.socket.send_json(headers)
                print("[out] IMAGE:", image.shape, image.dtype, np.max(image))
                self.socket.send(np.ascontiguousarray(image))
                print("[out] MASK:", mask.shape, mask.dtype, np.max(image))
                self.socket.send(np.ascontiguousarray(mask))
                print("[out] LABELS:", labels.shape, labels.dtype, np.max(labels) )
                self.socket.send(np.ascontiguousarray(labels))
                num += 1

                print()
                print()


            print("%s: %d/%d message block sent... " % (self.name, num, len(keys)))


        # will be never called actually
        self.h5.close()


    def read_data(self, key):

        entry = self.datum[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        meta = json.loads(entry.attrs['meta'])
        meta['joints'] = RmpeCocoConfig.convert(np.array(meta['joints']))
        data = entry.value

        if data.shape[0] <= 6:
            # TODO: this is extra work, should write in store in correct format (not transposed)
            # can't do now because I want storage compatibility yet
            # we need image in classical not transposed format in this program for warp affine
            data = data.transpose([1,2,0])

        img = data[:,:,0:3]
        mask_miss = data[:,:,4]
        mask = data[:,:,5]

        return img, mask_miss, meta


    def produce_headers(self, img, mask, labels):

        header_data = {"descr": img.dtype.str, "shape": img.shape,    "fortran_order": False}
        header_mask = {"descr": mask.dtype.str, "shape": mask.shape,   "fortran_order": False}
        header_label = {"descr": labels.dtype.str,  "shape": labels.shape, "fortran_order": False}
        headers = [header_data, header_mask, header_label]

        return headers

    def transform_data(self, img, mask,  meta):

        img, mask, meta = Transformer.transform(img, mask, meta)
        labels = self.heatmapper.create_heatmaps(meta['joints'])
        return img, mask, meta, labels


def main():

    #train = Server("../dataset/train_dataset.h5", 5555, "Train")
    val = Server("../dataset/val_dataset.h5", 5556, "Val")

    processes = [val] #, train

    while None in [p.process.exitcode for p in processes]:

        print("exitcodes", [p.process.exitcode for p in processes])
        for p in processes:
            if p.process.exitcode is None:
                p.join()


np.set_printoptions(precision=3, linewidth=75*3, suppress=True, threshold=100000)
main()


