#!/usr/bin/env python
import sys
import numpy as np
import zmq

from multiprocessing import Process
from py_rmpe_data_iterator import DataIterator

from time import time

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

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.set_hwm(1) #TODO: put 160 in production, made for debug purposes
        self.socket.bind("tcp://*:%s" % self.port)

    @staticmethod
    def loop(self):

        print("%s: Child process init... " % self.name)
        self.init()

        iterator = DataIterator(self.h5file, shuffle=False, augment=True)

        print("%s: Loop started... " % self.name)

        num = 0
        generation = 0
        cycle_start = time()

        while True:


            keys = iterator.num_keys()
            print("%s: generation %s, %d images " % (self.name, generation, keys))

            start = time()
            for (image, mask, labels, debug) in iterator.iterate():

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
                print("[%d/%d] %0.2f images per second (last image %0.2f ms)" % (num, keys, num/(time() - cycle_start), (time() - start)*1000 ) )
                start = time()

                print()
                print()


    def produce_headers(self, img, mask, labels):

        header_data = {"descr": img.dtype.str, "shape": img.shape, "fortran_order": False}
        header_mask = {"descr": mask.dtype.str, "shape": mask.shape,   "fortran_order": False}
        header_label = {"descr": labels.dtype.str,  "shape": labels.shape, "fortran_order": False}
        headers = [header_data, header_mask, header_label]

        return headers


def main():

    train = Server("../dataset/train_dataset.h5", 5555, "Train")
    val = Server("../dataset/val_dataset.h5", 5556, "Val")

    processes = [val, train] #,

    while None in [p.process.exitcode for p in processes]:

        print("exitcodes", [p.process.exitcode for p in processes])
        for p in processes:
            if p.process.exitcode is None:
                p.join()


np.set_printoptions(precision=1, linewidth=75*3, suppress=True, threshold=100000)
main()


