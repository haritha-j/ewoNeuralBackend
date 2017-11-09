#!/usr/bin/env python

import sys
import os
sys.path.append("..")

from time import time
from training.ds_generator_client import DataGeneratorClient

import cv2
import numpy as np


def save_image(num, type, img):

    if len(img.shape) == 3 and img.shape[0] == 57:
        #this is heatmaps
        img = np.zeros(img.shape[0],)

    if len(img.shape) == 3 and img.shape[0] == 3:
        img = img.transpose([1,2,0])

    if len(img.shape)==2:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    if img.dtype == np.float:
        img = img * 255
        img = img.astype(np.uint8)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite("output/%07d%s.png" % (num, type), img)


def time_processed(client, batch_size):

    num = 0
    start = time()

    for x,y in client.gen():
        num += 1
        elapsed = time() - start
        print(num*batch_size, num*batch_size/elapsed, [ i.shape for i in x ], [i.shape for i in y] )

def time_raw(client, save):

    num = 0
    start = time()

    for x,y,z in client.gen_raw():
        num += 1
        elapsed = time() - start
        print(num, num/elapsed, x.shape, y.shape, z.shape )

        if save:
            save_image(num, '', x)
            save_image(num, 'mask', y)
            save_image(num, 'parts', z)


def main(type, batch_size, save):

    client = DataGeneratorClient(port=5556, host="localhost", hwm=1, batch_size=batch_size)
    client.restart()

    if type=='processed':
        time_processed(client, batch_size)
    elif type=='raw':
        time_raw(client, save)
    else:
        assert False, "type should be 'processed' or 'raw' "


assert len(sys.argv) >=2,  "Usage: ./rmpe_dataset_server_stress_tester <processed|raw> [batch_size] [save]"
batch_size=10
save = False
if 'save' in sys.argv:
    save=True
    sys.argv = [s for s in sys.argv if s!='save']
if len(sys.argv)==3: batch_size=int(sys.argv[2])

main(sys.argv[1], batch_size, save)
