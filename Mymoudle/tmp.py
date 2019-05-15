# -*-encoding:utf-8-*-
import tensorflow as tf
import numpy as np
import random

import os
from SVD.static import *

'''
count = 0
for i in range(5):
    for j in range(5):
        if j > 3:
            break
        else:
            count += 1
            p = (i+1)*(j+1)
            print(p)
'''

cur_list = []
if len(cur_list) == 0:
    print("yes")


result = np.load(os.path.join(data_dir, 'server', 'result.npy'))
