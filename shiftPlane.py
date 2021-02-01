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

def shiftPlane(flowPath,csvPath):
    gridSize=64

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
                (nx,ny,nz) = (float(row[6]), float(row[7]), float(row[8]))
                # (nx,ny,nz) = (nx*sizeX/128.0,ny*sizeY/128.0,nz*sizeZ/128.0)
                # s = sqrt(nx**2+ny**2+nz**2)
                s = float(row[5])
                # (nx,ny,nz) = (nx/s,ny/s,nz/s)
                s = s*(max(sizeX,sizeY,sizeZ)/128.0)
                # print('Center = ({},{},{})'.format(x0,y0,z0))
                # print('Normal = ({},{},{})'.format(nx,ny,nz))
                # print('Side = {}'.format(s))

                (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.array((nx,ny,nz)))

                magP = np.load(flowPath + '/' + vessel + '_MAG.npy')
                cdP = np.load(flowPath + '/' + vessel + '_CD.npy')
                velNormalP = np.load(flowPath + '/' + vessel + '_velNormal.npy')
                meanVelP = np.mean(velNormalP,axis=2)
                (nxP,nyP) = cdP.shape
                (i, j) = np.meshgrid(np.linspace(-1.0,1.0,nxP), np.linspace(-1.0,1.0,nyP))
                # plt.figure()
                # plt.imshow(meanVelP)
                # plt.title('meanVel')
                # plt.show()
                blur = cv2.blur(meanVelP,(6,6))*cv2.blur(cdP,(6,6))
                # plt.figure()
                # plt.imshow(cdP)
                # plt.title('cd')
                # plt.show()
                ind = np.argmax(blur)
                # print('{},{}'.format(i.ravel()[ind],j.ravel()[ind]))

                x0 = x0+x1*j.ravel()[ind]*s/2.0*0.75 + y1*i.ravel()[ind]*s/2.0*0.75
                y0 = y0+x2*j.ravel()[ind]*s/2.0*0.75 + y2*i.ravel()[ind]*s/2.0*0.75
                z0 = z0+x3*j.ravel()[ind]*s/2.0*0.75 + y3*i.ravel()[ind]*s/2.0*0.75
                xi1 = int(max(0,round(x0-2)))
                xi2 = int(min(sizeX,round(x0+3)))
                yi1 = int(max(0,round(y0-2)))
                yi2 = int(min(sizeY,round(y0+3)))
                zi1 = int(max(0,round(z0-2)))
                zi2 = int(min(sizeZ,round(z0+3)))
                vXM = np.mean(vX[xi1:xi2,yi1:yi2,zi1:zi2,:].astype(np.float32),axis=3)
                vYM = np.mean(vY[xi1:xi2,yi1:yi2,zi1:zi2,:].astype(np.float32),axis=3)
                vZM = np.mean(vZ[xi1:xi2,yi1:yi2,zi1:zi2,:].astype(np.float32),axis=3)
                # print('Pre = {},{},{}'.format(nx,ny,nz))
                nx2=3.0*np.mean(vXM)+nx
                ny2=3.0*np.mean(vYM)+ny
                nz2=3.0*np.mean(vZM)+nz
                l=sqrt((nx2)**2+(ny2)**2+(nz2)**2)
                nx=nx2/l
                ny=ny2/l
                nz=nz2/l

                with open(csvPath.replace('.csv','_shifted.csv'), mode='a') as csvFile2:
                    writer = csv.writer(csvFile2, delimiter=',')
                    writer.writerow([row[0],row[1],x0,y0,z0,s,nx,ny,nz])
                # print('Post = {},{},{}'.format(nx,ny,nz))
                (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.array((nx,ny,nz)))

                (minX,maxX,minY,maxY,minZ,maxZ) = getPlaneBounds((x0,y0,z0),(x1,x2,x3),(y1,y2,y3),(sizeX,sizeY,sizeZ),s)

                (i, j) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize), np.linspace(start=-s/2.0, stop=s/2.0, num=gridSize))

                xp = x0 + x1*i + y1*j
                yp = y0 + x2*i + y2*j
                zp = z0 + x3*i + y3*j

                # print('Plane Query bounds:')
                # print(min(xp.flatten()),max(xp.flatten()),min(yp.flatten()),max(yp.flatten()),min(zp.flatten()),max(zp.flatten()))

                for (name,img) in [(vessel+'_CD',cd),(vessel+'_MAG',mag)]:
                    # print(img.shape)

                    rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
                                                   np.linspace(minY, maxY, maxY-minY+1),
                                                   np.linspace(minZ, maxZ, maxZ-minZ+1)),
                                                   img[minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1)],
                                                   bounds_error=False, fill_value=0)

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
                        grid = np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(gridSize,gridSize)))
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
    shiftPlane(flowPath,csvPath)
