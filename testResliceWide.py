import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
import sys
import os
import csv
import reslicePlane
import segmentPlane

csvFile = '/export/home/pcorrado/CODE/STUDIES/DL_Plane_Placement/cutPlaneList.csv'
dirName = '/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2017_Eldridge_NLP_MET/031/Normoxia/CS_WAVELET_20_FRAMES_LOWRES_WIDE/dat/CORRECTED'
reslicePlane.reslicePlane(dirName,csvFile)
segmentPlane.segmentPlane(dirName+'/Aorta_MAG.npy')
segmentPlane.segmentPlane(dirName+'/MPA_MAG.npy')
segmentPlane.segmentPlane(dirName+'/SVC_MAG.npy')
segmentPlane.segmentPlane(dirName+'/IVC_MAG.npy')
