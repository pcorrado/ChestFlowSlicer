import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
from scipy.interpolate import RegularGridInterpolator
import random
import warnings
import sys
import os
import csv

def is_perfect_cube(x):
    x = abs(x)
    return int(round(x ** (1. / 3))) ** 3 == x

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
    # print('Plane Bounds = ')
    # print(minX,maxX,minY,maxY,minZ,maxZ)
    return (minX,maxX,minY,maxY,minZ,maxZ)

def loadDat(fileName,size,minimum=None,maximum=None):
    img = np.fromfile(fileName, dtype=np.int16)
    img = img.reshape(size)
    if not (minimum is None):
        img = np.maximum(np.zeros(img.shape),img - minimum)
    if not (maximum is None):
        img = np.minimum(img,np.ones(img.shape)*maximum)
    # plt.hist(self.data.flatten().tolist(), bins=100)
    # plt.show()
    return np.transpose(img,axes=(2,1,0))

def reslicePlane(flowPath,csvPath):
    gridSize=64
    # print('Input flow path = {}'.format(flowPath))
    if not os.path.exists(flowPath):
        flowPath = '\\\\MPUFS7\\data_mrcv\\45_DATA_HUMANS\\CHEST\\STUDIES\\2017_Eldridge_NLP_MET\\002\\Normoxia\\CS_WAVELET_20_FRAMES_LOWRES\\dat\\CORRECTED'
        print('Flow Path not found, using default instead.')
    # print('4D flow path = {}'.format(flowPath))
    # print('Input CSV path = {}'.format(csvPath))
    if not os.path.exists(csvPath):
        csvPath = 'M:\\pcorrado\\CODE\\chestFlowSlicer\\testCSVSaveFile.csv'
        print('CSV Path not found, using default instead.')
    # print('CSV path = {}'.format(csvPath))

    headerFile = os.path.join(flowPath,'pcvipr_header.txt')
    for line in open(headerFile,'r'):
        (field,value) = line.split()
        if field=='matrixx': sizeX=int(float(value))
        if field=='matrixy': sizeY=int(float(value))
        if field=='matrixz': sizeZ=int(float(value))
        if field=='frames': sizeT=int(float(value))

    # print('Size = ({},{},{},{})'.format(sizeX,sizeY,sizeZ,sizeT))
    mag = loadDat(os.path.join(flowPath,'MAG.dat'),[sizeZ,sizeY,sizeX])
    cd = loadDat(os.path.join(flowPath,'CD.dat'),[sizeZ,sizeY,sizeX],minimum=0)
    vX = np.zeros([sizeX,sizeY,sizeZ,sizeT])
    vY = np.zeros([sizeX,sizeY,sizeZ,sizeT])
    vZ = np.zeros([sizeX,sizeY,sizeZ,sizeT])
    for ph in range(sizeT):
        vxFile = 'ph_{:0>3d}_vd_{}.dat'.format(ph,1)
        vyFile = 'ph_{:0>3d}_vd_{}.dat'.format(ph,2)
        vzFile = 'ph_{:0>3d}_vd_{}.dat'.format(ph,3)
        vX[:,:,:,ph] = -loadDat(os.path.join(flowPath,vxFile),[sizeZ,sizeY,sizeX])
        vY[:,:,:,ph] = -loadDat(os.path.join(flowPath,vyFile),[sizeZ,sizeY,sizeX])
        vZ[:,:,:,ph] = -loadDat(os.path.join(flowPath,vzFile),[sizeZ,sizeY,sizeX])

    with open(csvPath, mode='r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if row and ((row[0]==flowPath) or (row[0].replace('\\\\MPUFS7','\\data').replace('\\','/')==flowPath)):
                print(row[0].replace('\\\\MPUFS7','\\data').replace('\\','/'))
                vessel = row[1]
                (x0,y0,z0) = (float(row[2]), float(row[3]), float(row[4]))
                (x0,y0,z0) = (x0*(sizeX/128.0),y0*(sizeY/128.0),z0*(sizeZ/128.0))
                (nx,ny,nz) = (float(row[5]), float(row[6]), float(row[7]))
                # (nx,ny,nz) = (nx*sizeX/128.0,ny*sizeY/128.0,nz*sizeZ/128.0)
                s = sqrt(nx**2+ny**2+nz**2)
                # s = float(row[5])
                (nx,ny,nz) = (nx/s,ny/s,nz/s)
                s = s*(max(sizeX,sizeY,sizeZ)/128.0)
                # print('Center = ({},{},{})'.format(x0,y0,z0))
                # print('Normal = ({},{},{})'.format(nx,ny,nz))
                # print('Side = {}'.format(s))

                (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.array((nx,ny,nz)))
                (minX,maxX,minY,maxY,minZ,maxZ) = getPlaneBounds((x0,y0,z0),(x1,x2,x3),(y1,y2,y3),(sizeX,sizeY,sizeZ),s)

                # print('x=({},{},{})'.format(x1,x2,x3))
                # print('y=({},{},{})'.format(y1,y2,y3))
                (i, j) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize), np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize))

                # print('IJ Bounds')
                # print(min(i.flatten()),max(i.flatten()),min(j.flatten()),max(j.flatten()))
                # xp = np.minimum(np.maximum(x0 + i*x1 + j*y1,np.zeros(i.shape)),np.ones(i.shape)*(sizeX-1))
                # yp = np.minimum(np.maximum(y0 + i*x2 + j*y2,np.zeros(i.shape)),np.ones(i.shape)*(sizeY-1))
                # zp = np.minimum(np.maximum(z0 + i*x3 + j*y3,np.zeros(i.shape)),np.ones(i.shape)*(sizeZ-1))
                # print(x0,x1,y1)
                xp = x0 + x1*i + y1*j
                yp = y0 + x2*i + y2*j
                zp = z0 + x3*i + y3*j

                # print('Plane Query bounds:')
                # print(min(xp.flatten()),max(xp.flatten()),min(yp.flatten()),max(yp.flatten()),min(zp.flatten()),max(zp.flatten()))

                for (name,img) in [(vessel+'_MAG',mag),(vessel+'_CD',cd)]:
                    # print(img.shape)
                    rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
                                                   np.linspace(minY, maxY, maxY-minY+1),
                                                   np.linspace(minZ, maxZ, maxZ-minZ+1)),
                                                   img[minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1)],
                                                   bounds_error=False, fill_value=0)

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
                        grid = np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(gridSize,gridSize)))
                    # plt.figure()
                    # plt.imshow(grid)
                    # plt.title(name)
                    np.save(os.path.join(flowPath,name),grid)


                grid =np.zeros((gridSize,gridSize,sizeT))
                for (v,n) in [(vX,nx),(vY,ny),(vZ,nz)]:
                    for ph in range(sizeT):

                        rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
                                                       np.linspace(minY, maxY, maxY-minY+1),
                                                       np.linspace(minZ, maxZ, maxZ-minZ+1)),
                                                       v[minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1),ph],
                                                       bounds_error=False, fill_value=0)

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
                            # if ph==3:
                            #     plt.figure()
                            #     plt.imshow(np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(gridSize,gridSize))))
                            #     plt.title('{},{}'.format(n,ph))
                            #     plt.show()
                            grid[:,:,ph] += n*np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(gridSize,gridSize)))
                np.save(os.path.join(flowPath,vessel+'_velNormal'),grid)

if __name__ == "__main__":
    flowPath = sys.argv[1]
    csvPath = sys.argv[2]
    reslicePlane(flowPath,csvPath)
