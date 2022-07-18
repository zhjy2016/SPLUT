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

LR_G = 1e-3 
OUT_NUM=4
VERSION = "SPLUT_L_0.001_4"

UPSCALE = 4     # upscaling factor
# Load LUT
LUTA1_122   = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,1 ,122)).astype(np.float32)

LUTA2_221_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,21,221)).astype(np.float32)
LUTA2_221_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,22,221)).astype(np.float32)
LUTA2_212_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,21,212)).astype(np.float32)
LUTA2_212_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,22,212)).astype(np.float32)

LUTA3_221_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,31,221)).astype(np.float32)
LUTA3_221_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,32,221)).astype(np.float32)
LUTA3_212_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,31,212)).astype(np.float32)
LUTA3_212_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,32,212)).astype(np.float32)

LUTB1_122   = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,1 ,122)).astype(np.float32)

LUTB2_221_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,21,221)).astype(np.float32)
LUTB2_221_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,22,221)).astype(np.float32)
LUTB2_212_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,21,212)).astype(np.float32)
LUTB2_212_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,22,212)).astype(np.float32)

LUTB3_221_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,31,221)).astype(np.float32)
LUTB3_221_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,32,221)).astype(np.float32)
LUTB3_212_1 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,31,212)).astype(np.float32)
LUTB3_212_2 = np.load("../training_testing_code/transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,32,212)).astype(np.float32)

LUTs = {}
LUTs[0] = LUTA1_122
LUTs[1] = LUTA2_221_1
LUTs[2] = LUTA2_221_2
LUTs[3] = LUTA2_212_1
LUTs[4] = LUTA2_212_2
LUTs[5] = LUTA3_221_1
LUTs[6] = LUTA3_221_2
LUTs[7] = LUTA3_212_1
LUTs[8] = LUTA3_212_2
LUTs[9] = LUTB1_122
LUTs[10] = LUTB2_221_1
LUTs[11] = LUTB2_221_2
LUTs[12] = LUTB2_212_1
LUTs[13] = LUTB2_212_2
LUTs[14] = LUTB3_221_1
LUTs[15] = LUTB3_221_2
LUTs[16] = LUTB3_212_1
LUTs[17] = LUTB3_212_2


names = ['LUTA1_122', 'LUTA2_221_1', 'LUTA2_221_2', 'LUTA2_212_1', 'LUTA2_212_2', 
        'LUTA3_221_1', 'LUTA3_221_2', 'LUTA3_212_1', 'LUTA3_212_2',
        'LUTB1_122', 'LUTB2_221_1', 'LUTB2_221_2', 'LUTB2_212_1', 'LUTB2_212_2', 
        'LUTB3_221_1', 'LUTB3_221_2', 'LUTB3_212_1', 'LUTB3_212_2',]

for i in range(18):
    f = open(f'luts/{names[i]}.csv','w',encoding='utf-8')
    cw = csv.writer(f)
    for j in range(LUTs[i].shape[0]):
        s = []
        for k in range(LUTs[i].shape[1]):
            s.append('%.4f'%LUTs[i][j][k][0][0])
        cw.writerow(s)
    f.close()
