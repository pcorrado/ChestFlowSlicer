#!/usr/bin/env python

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import ImageVolume as iv
import SliceViewer as sv
import CutPlane as cp
import CutPlaneViewer as cpv
from math import sqrt
import os
import csv
import stat

def normalizeLength(x,y,z):
    len = sqrt(float(x**2 + y**2 ++ z**2))
    if len>0: (x,y,z)=(x/len, y/len, z/len)
    return (x,y,z)
def normalizeXYLength(x,y,z):
    len = sqrt(float(x**2 + y**2))
    zFac = sqrt(1-z**2)
    if len>0: (x,y,z)=(x/len, y/len, z)
    return (x*zFac,y*zFac,z)
def normalizeXZLength(x,y,z):
    len = sqrt(float(x**2 + z**2))
    yFac = sqrt(1-y**2)
    if len>0: (x,y,z)=(x/len, y, z/len)
    return (x*yFac,y,z*yFac)
def normalizeYZLength(x,y,z):
    len = sqrt(float(y**2 + z**2))
    xFac = sqrt(1-x**2)
    if len>0: (x,y,z)=(x, y/len, z/len)
    return (x,y*xFac,z*xFac)

class PlaneComparer:
    def __init__(self,
                 root=tk.Tk(),
                 cdFile='\\\\MPUFS7\\data_mrcv\\45_DATA_HUMANS\\CHEST\\STUDIES\\2018_RV_FUNCTION_GOSS\\RVF_009\\DAY2\\02314_00023_PCVIPR\\CS_WAVELET_20_FRAMES_LOWRES\\dat\\CORRECTED\\CD.dat',
                 mode='destroy',
                 extraString=''):

        self._cutPlaneGridSize = 64
        self._targetSize=(128,128,128)

        self.cdFile=cdFile
        self.mode=mode

        self.root = root
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        height=0.86*screen_height
        width = 1.1*height
        top = (screen_height-height)/3
        left = (screen_width-width)/2
        self.root.geometry('{:d}x{:d}+{:d}+{:d}'.format(int(width),int(height),int(left),int(top)))
        self.root.title('Chest Flow Slicer: Quick plane placement for 4D flow MRI of the chest. {}'.format(extraString))
        self.imageVolume = iv.ImageVolume(cdFile,self._targetSize)

        tk.Grid.rowconfigure(self.root, 0, weight=1)
        tk.Grid.columnconfigure(self.root, 0, weight=1)

        self.frame=tk.Frame(self.root)
        self.frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self.frame, 0, weight=0)
        tk.Grid.rowconfigure(self.frame, 1, weight=1)
        tk.Grid.rowconfigure(self.frame, 2, weight=1)
        tk.Grid.columnconfigure(self.frame, 0, weight=1)
        tk.Grid.columnconfigure(self.frame, 1, weight=1)

        img = self.imageVolume.getAxialImageSlice(round(self.imageVolume.nz/2))
        self.axialSliceViewer = sv.SliceViewer(self.frame,img=img,row=1,col=1,nx=self.imageVolume.nx, ny=self.imageVolume.ny, nSlice=self.imageVolume.nz)
        img = self.imageVolume.getSagittalImageSlice(round(self.imageVolume.nx/2))
        self.sagittalSliceViewer = sv.SliceViewer(self.frame,img=img,row=1,col=0,nx=self.imageVolume.ny, ny=self.imageVolume.nz,nSlice=self.imageVolume.nx)
        img = self.imageVolume.getCoronalImageSlice(round(self.imageVolume.ny/2))
        self.coronalSliceViewer = sv.SliceViewer(self.frame,img=img,row=2,col=0,nx=self.imageVolume.nx, ny=self.imageVolume.nz,nSlice=self.imageVolume.ny)

        self.axialSliceViewer.setSliderCallback(self.setZ)
        self.sagittalSliceViewer.setSliderCallback(self.setX)
        self.coronalSliceViewer.setSliderCallback(self.setY)


        self.cutPlaneViewer = cpv.CutPlaneViewer(self.frame,
            img=Image.fromarray(np.zeros((self._cutPlaneGridSize,self._cutPlaneGridSize))),
            row=2,
            col=1)

        self.cutPlanes = [cp.CutPlane(x=70.68439232, y=59.68111969,	z=54.30117627, s=12, nx=0.039242356, ny=-0.706362402, nz=-0.706761767),
                          cp.CutPlane(x=73.1,	y=56.7,	z=54.9,	s=15.8,	nx=-0.15, ny=-0.59,	nz=-0.79)]

        self.planeInitialized = [[False,False,False] for _ in self.cutPlanes]

        self.setX(round(self.imageVolume.nx/2))
        self.setY(round(self.imageVolume.ny/2))
        self.setZ(round(self.imageVolume.nz/2))



        self.updateUI()

        root.mainloop()

    def setZ(self,val):
        self.z = int(val)
        self.giveNewAxialSlice(val)
    def setX(self,val):
        self.x = int(val)
        self.giveNewSagittalSlice(val)
    def setY(self,val):
        self.y = int(val)
        self.giveNewCoronalSlice(val)

    def giveNewAxialSlice(self,slice):
        self.axialSliceViewer.setImage(self.imageVolume.getAxialImageSlice(int(slice)))
        self.axialSliceViewer.scale.set(int(slice))
        self.sagittalSliceViewer.setYCrossing(slice)
        self.coronalSliceViewer.setYCrossing(slice)
        self.giveAxialIntersection()

    def giveNewSagittalSlice(self,slice):
        self.sagittalSliceViewer.setImage(self.imageVolume.getSagittalImageSlice(int(slice)))
        self.sagittalSliceViewer.scale.set(int(slice))
        self.coronalSliceViewer.setXCrossing(slice)
        self.axialSliceViewer.setXCrossing(slice)
        self.giveSagittalIntersection()

    def giveNewCoronalSlice(self,slice):
        self.coronalSliceViewer.setImage(self.imageVolume.getCoronalImageSlice(int(slice)))
        self.coronalSliceViewer.scale.set(int(slice))
        self.axialSliceViewer.setYCrossing(slice)
        self.sagittalSliceViewer.setXCrossing(slice)
        self.giveCoronalIntersection()


    def updateUI(self):
        self.axialSliceViewer.updateUI()
        self.sagittalSliceViewer.updateUI()
        self.coronalSliceViewer.updateUI()
        self.cutPlaneViewer.updateUI()

    def giveInersection(self, sliceViewer, normal,line, ind, sv,lineCallback=None, arrowCallback=None):
        if self.planeInitialized[ind][sv]:
            sliceViewer.movePlane(normal=normal, line=line, index=ind)
        else:
            sliceViewer.addPlane2(line,normal,None,None, index=ind)
            self.planeInitialized[ind][sv] = True

    def giveAxialIntersection(self):
        for i,plane in enumerate(self.cutPlanes):
            (nx,ny,_) = plane.normal
            (x1,y1,x2,y2) = plane.calcZIntersection(self.z)
            self.giveInersection(self.axialSliceViewer,(nx,ny),(x1,y1,x2,y2),i,0)
    def giveSagittalIntersection(self):
        for i,plane in enumerate(self.cutPlanes):
            (_,ny,nz) = plane.normal
            (x1,y1,x2,y2) = plane.calcXIntersection(self.x)
            self.giveInersection(self.sagittalSliceViewer,(ny,nz),(x1,y1,x2,y2),i,1)
    def giveCoronalIntersection(self):
        for i,plane in enumerate(self.cutPlanes):
            (nx,_,nz) = plane.normal
            (x1,y1,x2,y2) = plane.calcYIntersection(self.y)
            self.giveInersection(self.coronalSliceViewer,(nx,nz),(x1,y1,x2,y2),i,2)


if __name__ == '__main__':
    app = PlaneComparer()
