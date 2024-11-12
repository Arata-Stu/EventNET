import numpy as np

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
DTYPE = np.dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4'), ('img_name', '<U10')])