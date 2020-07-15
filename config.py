import os
import numpy as np

image_dir = './imgs'
MRT = 0.7# 特征点匹配的参数，越大匹配越严格
K = np.array([
        [2362.12, 0, 1936/2],
        [0, 2362.12,  1296/2],
        [0, 0, 1]])

#集束调整中选择性删除所选点的范围。
x = 0.5
y = 1