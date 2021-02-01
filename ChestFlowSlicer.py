#!/usr/bin/env python

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import ChestFlowSlicer.ImageVolume as iv
import ChestFlowSlicer.SliceViewer as sv
import ChestFlowSlicer.CutPlane as cp
import ChestFlowSlicer.CutPlaneViewer as cpv
from math import sqrt
import os
import csv
import stat
import time
import getpass

def get_username():
    return os.getlogin()
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

class ChestFlowSlicer:
    def __init__(self,
                 root=tk.Tk(),
                 cdFile='\\\\MPUFS7\\data_mrcv\\45_DATA_HUMANS\\CHEST\\STUDIES\\2017_Eldridge_NLP_MET\\021\\Normoxia\\CS_WAVELET_20_FRAMES_LOWRES\\dat\\CORRECTED\\CD.dat',
                 saveCSVFile='M:\\pcorrado\\CODE\\ChestFlowSlicer\\testCSVSaveFile.csv',
                 mode='destroy',
                 extraString='',
                 vessels=[]):

        self._cutPlaneGridSize = 64
        self._targetSize=(128,128,128)

        self.cdFile=cdFile
        self.saveCSVFile = saveCSVFile
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

        self.buttonFrame = tk.Frame(master=self.frame)
        self.buttonFrame.grid(row=0,column=0,columnspan=2,sticky=tk.E+tk.W)
        self.vessels = [("Aorta",0),("MPA",1),("SVC",2),("IVC",3),("RSPV",4),("RIPV",5),("LSPV",6),("LIPV",7)]

        self.v = tk.IntVar()
        self.v.set(0)  # initializing the choice, i.e. Aorta

        self.vesselLabel = tk.Label(self.buttonFrame, text="Vessel",justify = tk.LEFT).pack(side=tk.LEFT,fill=tk.X, expand=tk.NO)
        self.radioButtons = [tk.Radiobutton(self.buttonFrame,text=vessel,variable=self.v,value=val, command=self.setActiveVessel)
                            for (vessel,val) in self.vessels]
        for button in self.radioButtons:
            button.pack(side=tk.LEFT,fill=tk.X, expand=tk.YES)

        self.saveButton = tk.Button(self.buttonFrame, text="Save", justify = tk.LEFT, command=self.saveButtonCallback)
        self.saveButton.pack(side=tk.LEFT,fill=tk.X, expand=tk.YES)
        self.doneButton = tk.Button(self.buttonFrame, text="Done", justify = tk.LEFT, command=self.doneButtonCallback)
        self.doneButton.pack(side=tk.LEFT,fill=tk.X, expand=tk.YES)

        img = self.imageVolume.getAxialImageSlice(round(self.imageVolume.nz/2))
        self.axialSliceViewer = sv.SliceViewer(self.frame,img=img,row=1,col=1,nx=self.imageVolume.nx, ny=self.imageVolume.ny, nSlice=self.imageVolume.nz)
        img = self.imageVolume.getSagittalImageSlice(round(self.imageVolume.nx/2))
        self.sagittalSliceViewer = sv.SliceViewer(self.frame,img=img,row=1,col=0,nx=self.imageVolume.ny, ny=self.imageVolume.nz,nSlice=self.imageVolume.nx)
        img = self.imageVolume.getCoronalImageSlice(round(self.imageVolume.ny/2))
        self.coronalSliceViewer = sv.SliceViewer(self.frame,img=img,row=2,col=0,nx=self.imageVolume.nx, ny=self.imageVolume.nz,nSlice=self.imageVolume.ny)

        self.axialSliceViewer.setSliderCallback(self.setZ)
        self.sagittalSliceViewer.setSliderCallback(self.setX)
        self.coronalSliceViewer.setSliderCallback(self.setY)
        self.axialSliceViewer.setClickCallback(self.axialClick)
        self.sagittalSliceViewer.setClickCallback(self.sagittalClick)
        self.coronalSliceViewer.setClickCallback(self.coronalClick)

        self.cutPlaneViewer = cpv.CutPlaneViewer(self.frame,
            img=Image.fromarray(np.zeros((self._cutPlaneGridSize,self._cutPlaneGridSize))),
            row=2,
            col=1,
            plusCallback=self.growPlaneCallback,
            minusCallback=self.shrinkPlaneCallback)

        self.setX(round(self.imageVolume.nx/2))
        self.setY(round(self.imageVolume.ny/2))
        self.setZ(round(self.imageVolume.nz/2))

        self.cutPlanes = [cp.CutPlane(x=self.x, y=self.y, z=self.z) for (vessel, vesselNum) in self.vessels]
        self.planeInitialized = [False for _ in self.cutPlanes]

        for (name,x,y,z,nx,ny,nz) in vessels:
            for (vessel, vesselNum) in self.vessels:
                if (name==vessel):
                    s = sqrt(nx**2 + ny**2 + nz**2)
                    self.v.set(vesselNum)
                    self.setPlaneLocation(center=(x,y,z), normal=(nx/s,ny/s,nz/s), side=s, redraw=True)
        self.v.set(0)

        self.updateUI()
        self.startTime = time.time()

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
        (_,vesselNum) = self.getActiveVessel()
        if hasattr(self,'planeInitialized') and self.planeInitialized[vesselNum]:
            self.giveAxialIntersection(vesselNum)

    def giveNewSagittalSlice(self,slice):
        self.sagittalSliceViewer.setImage(self.imageVolume.getSagittalImageSlice(int(slice)))
        self.sagittalSliceViewer.scale.set(int(slice))
        self.coronalSliceViewer.setXCrossing(slice)
        self.axialSliceViewer.setXCrossing(slice)
        (_,vesselNum) = self.getActiveVessel()
        if hasattr(self,'planeInitialized') and self.planeInitialized[vesselNum]:
            self.giveSagittalIntersection(vesselNum)

    def giveNewCoronalSlice(self,slice):
        self.coronalSliceViewer.setImage(self.imageVolume.getCoronalImageSlice(int(slice)))
        self.coronalSliceViewer.scale.set(int(slice))
        self.axialSliceViewer.setYCrossing(slice)
        self.sagittalSliceViewer.setXCrossing(slice)
        (_,vesselNum) = self.getActiveVessel()
        if hasattr(self,'planeInitialized') and self.planeInitialized[vesselNum]:
            self.giveCoronalIntersection(vesselNum)

    def axialClick(self,x,y):
        (_,val) = self.getActiveVessel()
        self.setX(x)
        self.setY(y)
        if not self.planeInitialized[val]:
            self.setPlaneLocation(center=(self.x,self.y,self.z), normal=(0.0001,0.0001,0.999), redraw=True)

    def sagittalClick(self,x,y):
        (_,val) = self.getActiveVessel()
        self.setY(x)
        self.setZ(y)
        if not self.planeInitialized[val]:
            self.setPlaneLocation(center=(self.x,self.y,self.z), normal=(0.9999,0.0001,0.0001), redraw=True)

    def coronalClick(self,x,y):
        (_,val) = self.getActiveVessel()
        self.setX(x)
        self.setZ(y)
        if not self.planeInitialized[val]:
            self.setPlaneLocation(center=(self.x,self.y,self.z), normal=(0.0001,0.9999,0.0001), redraw=True)

    def updateUI(self):
        self.axialSliceViewer.updateUI()
        self.sagittalSliceViewer.updateUI()
        self.coronalSliceViewer.updateUI()
        self.cutPlaneViewer.updateUI()

    def setActiveVessel(self):
        (vessel, val) = self.getActiveVessel()
        print('Active Vessel = {} {}'.format(vessel,val))
        if self.planeInitialized[val]:
            self.setPlaneLocation(redraw=True)
        else:
            self.cutPlaneViewer.setImage()
            for viewer in [self.axialSliceViewer, self.sagittalSliceViewer, self.coronalSliceViewer]:
                viewer.removePlane()

    def getActiveVessel(self):
        return self.vessels[int(self.v.get())]

    def saveButtonCallback(self):
        cdPath = os.path.dirname(self.cdFile)
        if not os.path.exists(self.saveCSVFile):
            with open(self.saveCSVFile, mode='w') as csvFile:
                writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Path', 'Vessel', 'Center_X', 'Center_Y', 'Center_Z', 'Normal_X', 'Normal_Y', 'Normal_Z','User','Processing Time'])
        else:
            (head,tail) = os.path.split(self.saveCSVFile)
            tempCSVFileName = os.path.join(head,'.'+tail)
            with open(tempCSVFileName, mode='w') as newCSVFile:
                with open(self.saveCSVFile, mode='r') as oldFile:
                    writer = csv.writer(newCSVFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    reader = csv.reader(oldFile, delimiter=',')
                    for row in reader:
                        if row and row[0]!=cdPath:
                            writer.writerow(row)
            os.remove(self.saveCSVFile)
            os.rename(tempCSVFileName,self.saveCSVFile)
            os.chmod(self.saveCSVFile, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        with open(self.saveCSVFile, mode='a') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for (vessel, val) in self.vessels:
                (nx,ny,nz) = self.cutPlanes[val].normal
                (cx,cy,cz) = self.cutPlanes[val].center
                s = self.cutPlanes[val].s
                writer.writerow([cdPath, vessel, cx, cy, cz, nx*s, ny*s, nz*s, get_username(), time.time()-self.startTime])
        print('Saved {}'.format(cdPath))

    def doneButtonCallback(self):
        self.saveButtonCallback()
        if self.mode=='quit':
            self.root.quit()
        else:
            self.root.destroy()

    def growPlaneCallback(self):
        self.changePlaneSize(2)
    def shrinkPlaneCallback(self):
        self.changePlaneSize(-2)
    def changePlaneSize(self,dS):
        (_,vesselNum) = self.getActiveVessel()
        self.setPlaneLocation(side=(self.cutPlanes[vesselNum].s+dS))

    def giveIntersection(self, vesselNum, sliceViewer, redraw, normal,line, lineCallback, arrowCallback):
            if redraw:
                sliceViewer.removePlane()
                sliceViewer.addPlane(normal=normal, line=line, lineCallback=lineCallback, arrowCallback=arrowCallback)
            else:
                sliceViewer.movePlane(normal=normal, line=line)

    def giveAxialIntersection(self,vesselNum, redraw=False):
        (nx,ny,_) = self.cutPlanes[vesselNum].normal
        (x1,y1,x2,y2) = self.cutPlanes[vesselNum].calcZIntersection(self.z)
        self.giveIntersection(vesselNum,self.axialSliceViewer,redraw,(nx,ny),(x1,y1,x2,y2),self.axialLineMoveCallback,self.axialArrowMoveCallback)
    def giveSagittalIntersection(self,vesselNum, redraw=False):
        (_,ny,nz) = self.cutPlanes[vesselNum].normal
        (x1,y1,x2,y2) = self.cutPlanes[vesselNum].calcXIntersection(self.x)
        self.giveIntersection(vesselNum,self.sagittalSliceViewer,redraw,(ny,nz),(x1,y1,x2,y2),self.sagittalLineMoveCallback,self.sagittalArrowMoveCallback)
    def giveCoronalIntersection(self,vesselNum, redraw=False):
        (nx,_,nz) = self.cutPlanes[vesselNum].normal
        (x1,y1,x2,y2) = self.cutPlanes[vesselNum].calcYIntersection(self.y)
        self.giveIntersection(vesselNum,self.coronalSliceViewer,redraw,(nx,nz),(x1,y1,x2,y2),self.coronalLineMoveCallback,self.coronalArrowMoveCallback)

    def axialLineMoveCallback(self,x,y):
        self.setPlaneLocation(center=(x,y,self.z))
    def sagittalLineMoveCallback(self,x,y):
        self.setPlaneLocation(center=(self.x,x,y))
    def coronalLineMoveCallback(self,x,y):
        self.setPlaneLocation(center=(x,self.y,y))

    def axialArrowMoveCallback(self,x,y):
        (_,vesselNum) = self.getActiveVessel()
        (cx,cy,cz) = self.cutPlanes[vesselNum].center
        (_,_,nz) = self.cutPlanes[vesselNum].normal
        self.setPlaneLocation(normal=normalizeXYLength(x-cx,y-cy,nz))
    def sagittalArrowMoveCallback(self,x,y):
        (_,vesselNum) = self.getActiveVessel()
        (cx,cy,cz) = self.cutPlanes[vesselNum].center
        (nx,_,_) = self.cutPlanes[vesselNum].normal
        self.setPlaneLocation(normal=normalizeYZLength(nx,x-cy,y-cz))
    def coronalArrowMoveCallback(self,x,y):
        (_,vesselNum) = self.getActiveVessel()
        (cx,cy,cz) = self.cutPlanes[vesselNum].center
        (_,ny,_) = self.cutPlanes[vesselNum].normal
        self.setPlaneLocation(normal=normalizeXZLength(x-cx,ny,y-cz))

    def scrollCallback(self,dx,dy,dz):
        (_,vesselNum) = self.getActiveVessel()
        (cx,cy,cz) = self.cutPlanes[vesselNum].center
        self.setPlaneLocation(center=(cx+dx,cy+dy,cz+dz))

    def setPlaneLocation(self, center=None, normal=None, side=None, redraw=False):
        (_,vesselNum) = self.getActiveVessel()
        if not (center is None):
            self.cutPlanes[vesselNum].center = center
        if not (normal is None):
            self.cutPlanes[vesselNum].normal = normal
        if not (side is None):
            self.cutPlanes[vesselNum].s = side
        self.giveAxialIntersection(vesselNum, redraw=redraw)
        self.giveSagittalIntersection(vesselNum, redraw=redraw)
        self.giveCoronalIntersection(vesselNum, redraw=redraw)
        self.planeInitialized[vesselNum] = True
        (img,xDir,yDir) = self.imageVolume.getObliqueSlice(center=self.cutPlanes[vesselNum].center,
                                                           normal=self.cutPlanes[vesselNum].normal,
                                                           s=self.cutPlanes[vesselNum].s,
                                                           gridSize=self._cutPlaneGridSize)
        self.cutPlaneViewer.setImage(img=img,xDir=xDir,yDir=yDir,scrollCallback=self.scrollCallback)

if __name__ == '__main__':
    app = ChestFlowSlicer()
