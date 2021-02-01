import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label
import random
import warnings
import sys
import os
import csv
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from matplotlib.colors import colorConverter
import matplotlib as mpl

def segmentPlane(filePath):
    # print('Input file path = {}'.format(filePath))
    if not os.path.exists(filePath):
        filePath = '\\\\MPUFS7\\data_mrcv\\45_DATA_HUMANS\\CHEST\\STUDIES\\2017_Eldridge_NLP_MET\\002\\Normoxia\\CS_WAVELET_20_FRAMES_LOWRES\\dat\\CORRECTED\\Aorta_MAG.npy'
        print('File not found, using default instead.')
    # print('File path = {}'.format(filePath))

    mag = np.load(filePath)
    cd = np.load(filePath.replace('_MAG','_CD'))
    velNormal = np.load(filePath.replace('_MAG','_velNormal'))
    meanVel = np.mean(velNormal,axis=2)

    seg = KMeans(n_clusters=2).fit(cd.reshape(cd.size,1)).labels_.reshape(mag.shape)

    (nx,ny) = cd.shape
    (i, j) = np.meshgrid(np.linspace(-1.0,1.0,nx), np.linspace(-1.0,1.0,ny))
    d2 = np.sqrt(np.sqrt(i*i + j*j))

    aveD1 = np.mean(d2*(seg==0))
    aveD2 = np.mean(d2*(seg==1))
    if aveD1<aveD2:
        seg = 1-seg
    #
    # color1 = colorConverter.to_rgba('white')
    # color2 = colorConverter.to_rgba('orange')
    # cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    # cmap2._init()
    # alphas = np.linspace(0, 0.2, cmap2.N+3)
    # cmap2._lut[:,-1] = alphas
    #
    # plt.figure()
    # plt.imshow(cd,cmap='gray')
    # plt.imshow(seg, cmap=cmap2)
    # plt.show()

    (a,b) = cv2.connectedComponents(seg.astype('uint8'))
    aveD = [0 for _ in range(a)]
    for reg in range(a):
        aveD[reg] = np.sum(d2*(b==reg))/np.sum(b==reg)

    seg = b==np.argmin(aveD)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    seg = cv2.dilate(seg.astype('uint8'),kernel=kernel)

    # color1 = colorConverter.to_rgba('white')
    # color2 = colorConverter.to_rgba('orange')
    # cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    # cmap2._init()
    # alphas = np.linspace(0, 0.2, cmap2.N+3)
    # cmap2._lut[:,-1] = alphas
    #
    # plt.figure()
    # plt.imshow(cd,cmap='gray')
    # plt.imshow(seg, cmap=cmap2)
    # plt.show()

    np.save(filePath.replace('_MAG','_SEG'),seg)

if __name__ == "__main__":
    filePath = sys.argv[1]
