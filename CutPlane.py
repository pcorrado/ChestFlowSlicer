import numpy as np
import tkinter as tk
from math import sqrt

def calcIntersection(sliceGap,nSlice,x,y,nx,ny,r):
    dSlice = abs(r*sqrt(1 - nSlice**2))
    if dSlice < sliceGap or dSlice<1.0:
        return (0,0,0,0)
    else:
        dx = r*ny*sqrt(1-sliceGap/dSlice)
        dy = r*nx*sqrt(1-sliceGap/dSlice)
        return (round(x-dx),round(y+dy),round(x+dx),round(y-dy))

class CutPlane:
    def __init__(self,x=0,y=0,z=0,nx=0,ny=0,nz=0,s=20):
        self.center = (x,y,z)
        self.normal = (nx,ny,nz)
        self.s = s

    def __str__(self):
        return 'Center = {}, Normal = {}, Side Length = {}'.format(self.center,self.normal,self.s)
        # print('Normal = {}'.format(self.normal))
        # print('Side Length = {}'.format(self.s))

    def calcXIntersection(self,xSlice):
        (x,y,z) = self.center
        (nx,ny,nz) = self.normal
        r = float(self.s)/2.0
        return calcIntersection(sliceGap=abs(xSlice-x),nSlice=nx,x=y,y=z,nx=ny,ny=nz,r=r)

    def calcYIntersection(self,ySlice):
        (x,y,z) = self.center
        (nx,ny,nz) = self.normal
        r = float(self.s)/2.0
        return calcIntersection(sliceGap=abs(ySlice-y),nSlice=ny,x=x,y=z,nx=nx,ny=nz,r=r)

    def calcZIntersection(self,zSlice):
        (x,y,z) = self.center
        (nx,ny,nz) = self.normal
        r = float(self.s)/2.0
        return calcIntersection(sliceGap=abs(zSlice-z),nSlice=nz,x=x,y=y,nx=nx,ny=ny,r=r)
