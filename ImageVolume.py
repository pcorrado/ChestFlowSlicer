import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import resample
import random
import warnings
import os

def getRowColumnVectors(n):
    random.seed(0)

    while True:
        x = np.array([random.random(),random.random(),random.random()])
        x = x-np.dot(x,n)*n
        if np.dot(x,x)>0:
            break
    x = x/sqrt(np.dot(x,x))

    while True:
        y = np.array([random.random(),random.random(),random.random()])
        y = y-np.dot(y,n)*n
        y = y-np.dot(y,x)*x
        if np.dot(y,y)>0:
            break
    y = y/sqrt(np.dot(y,y))

    if np.dot(np.cross(x,y),n)<0:
        y=-y

    return (x[0],x[1],x[2],y[0],y[1],y[2])

def getPlaneBounds(center,x,y,size,s):
    (x0,y0,z0) = center
    (x1,x2,x3) = x
    (y1,y2,y3) = y
    (sizeX,sizeY,sizeZ) = size
    minX = max(floor(x0 -  s/2.0*abs(x1) -  s/2.0*abs(y1)),0)
    maxX = min(ceil( x0 +  s/2.0*abs(x1) +  s/2.0*abs(y1)),sizeX-1)
    minY = max(floor(y0 -  s/2.0*abs(x2) -  s/2.0*abs(y2)),0)
    maxY = min(ceil( y0 +  s/2.0*abs(x2) +  s/2.0*abs(y2)),sizeY-1)
    minZ = max(floor(z0 -  s/2.0*abs(x3) -  s/2.0*abs(y3)),0)
    maxZ = min(ceil( z0 +  s/2.0*abs(x3) +  s/2.0*abs(y3)),sizeZ-1)
    return (minX,maxX,minY,maxY,minZ,maxZ)

def windowLevel(img):
    minimum = max(0,int(np.percentile(img.flatten(), 0.005)))
    maximum = int(np.percentile(img.flatten(), 99.8))
    return (img - minimum) * (255./maximum)

def subSample(img, targetSize):
    (tx,ty,tz) = targetSize
    img = resample(img,tx,axis=0)
    img = resample(img,ty,axis=1)
    return resample(img,tz,axis=2)

class ImageVolume:
    def __init__(self,fileName,targetSize):
        img = np.fromfile(fileName, dtype=np.int16)
        headerFile = os.path.join(os.path.dirname(fileName),'pcvipr_header.txt')
        for line in open(headerFile,'r'):
            (field,value) = line.split()
            if field=='matrixx': nx=int(float(value))
            if field=='matrixy': ny=int(float(value))
            if field=='matrixz': nz=int(float(value))
        img = img.reshape(nz,ny,nx)
        self.nx, self.ny, self.nz = targetSize

        self.data = windowLevel(subSample(np.transpose(img, axes=(2,1,0)),targetSize))

    def getAxialImageSlice(self,sliceNum):
        return Image.fromarray(np.transpose(np.squeeze(self.data[:,:,sliceNum])))

    def getSagittalImageSlice(self,sliceNum):
        return Image.fromarray(np.transpose(np.squeeze(self.data[sliceNum,:,:])))

    def getCoronalImageSlice(self,sliceNum):
        return Image.fromarray(np.transpose(np.squeeze(self.data[:,sliceNum,:])))

    def getObliqueSlice(self, center, normal, s, gridSize):
        (x0,y0,z0) = center
        (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.array(normal))
        (minX,maxX,minY,maxY,minZ,maxZ) = getPlaneBounds(center,(x1,x2,x3),(y1,y2,y3),self.data.shape,s)

        rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
                                       np.linspace(minY, maxY, maxY-minY+1),
                                       np.linspace(minZ, maxZ, maxZ-minZ+1)),
                                       self.data[minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1)],
                                       fill_value=0)
        (i, j) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize), np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize))

        (sizeX,sizeY,sizeZ) = self.data.shape
        xp = np.minimum(np.maximum(x0 + i*x1 + j*y1,np.zeros(i.shape)),np.ones(i.shape)*(sizeX-1))
        yp = np.minimum(np.maximum(y0 + i*x2 + j*y2,np.zeros(i.shape)),np.ones(i.shape)*(sizeY-1))
        zp = np.minimum(np.maximum(z0 + i*x3 + j*y3,np.zeros(i.shape)),np.ones(i.shape)*(sizeZ-1))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            grid = np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(gridSize,gridSize))
        return (Image.fromarray(grid),(x1,x2,x3),(y1,y2,y3))

    # def getNSlice(self):
    #     (nx,ny,nz) = self.data.shape
    #     return nz
