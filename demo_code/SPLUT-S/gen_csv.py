from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
from pdb import set_trace as st

import torch
import cv2
import sys
sys.path.insert(1, '../1_Train_deep_model')

import csv

LR_G = 1e-3         # Learning rate for the generator
OUT_NUM=4
VERSION = "SPLUT_S_0.001_4"

# USER PARAMS
UPSCALE = 4     # upscaling factor
# Load LUT
LUTA1_122 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,1,122)).astype(np.float32)
LUTA2_221 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,2,221)).astype(np.float32)
LUTA2_212 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,2,212)).astype(np.float32)
LUTA3_221 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,3,221)).astype(np.float32)
LUTA3_212 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,3,212)).astype(np.float32)

LUTB1_122 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,1,122)).astype(np.float32)
LUTB2_221 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,2,221)).astype(np.float32)
LUTB2_212 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,2,212)).astype(np.float32)
LUTB3_221 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,3,221)).astype(np.float32)
LUTB3_212 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,3,212)).astype(np.float32)

LUTs = {}
LUTs[0] = LUTA1_122
LUTs[1] = LUTA2_221
LUTs[2] = LUTA2_212
LUTs[3] = LUTA3_221
LUTs[4] = LUTA3_212
LUTs[5] = LUTB1_122
LUTs[6] = LUTB2_221
LUTs[7] = LUTB2_212
LUTs[8] = LUTB3_221
LUTs[9] = LUTB3_212

names = ['LUTA1_122', 'LUTA2_221', 'LUTA2_212', 'LUTA3_221', 'LUTA3_212', 'LUTB1_122', 'LUTB2_221', 'LUTB2_212', 'LUTB3_221', 'LUTB3_212']

for i in range(10):
    f = open(f'luts/{names[i]}.csv','w',encoding='utf-8')
    cw = csv.writer(f)
    for j in range(LUTs[i].shape[0]):
        s = []
        for k in range(LUTs[i].shape[1]):
            s.append('%.4f'%LUTs[i][j][k][0][0])
        cw.writerow(s)
    f.close()
