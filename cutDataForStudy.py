import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
import sys
import os
import csv
import reslicePlane
import segmentPlane
import calcFlow
from shiftPlane import shiftPlane

def doneYet(csvFile,path,vessel):
    if os.path.exists(csvFile):
        with open(csvFile,mode='r') as csvFileReader:
            reader = csv.reader(csvFileReader, delimiter=',')
            for row in reader:
                if row and row[0]==path and row[1]==vessel:
                    return True
    return False

def cutDataForStudy(csvFile,saveCSVFile):
    with open(csvFile,mode='r') as csvFileReader:
        reader = csv.reader(csvFileReader, delimiter=',')
        for row in reader:
            if row:
                dirName = row[0].replace('\\\\MPUFS7', '/data').replace('\\','/')
                print(dirName)
                if os.path.exists(dirName):
                    (nx,ny,nz) = (float(row[6]),float(row[7]),float(row[8]))
                    s = float(row[5])
                    magFile = os.path.join(dirName,'{}_MAG.npy'.format(row[1]))
                    # if not os.path.exists(magFile):
                    reslicePlane.reslicePlane(dirName,csvFile)
                    shiftPlane(dirName,csvFile) #TEMP!! REMOVE THIS LATER
                    # if not doneYet(saveCSVFile,dirName,row[1]):
                    #     segmentPlane.segmentPlane(magFile)
                    #     headerFile = os.path.join(dirName,'pcvipr_header.txt')
                    #     for line in open(headerFile,'r'):
                    #         (field,value) = line.split()
                    #         if field=='matrixx': sizeX=int(float(value))
                    #         if field=='fovx': fovX=float(value)
                    #         if field=='frames': nT=int(float(value))
                    #         if field=='timeres': dt=float(value)
                    #     dx = fovX/sizeX
                    #     RR = nT*dt
                    #     calcFlow.calcFlow(os.path.join(dirName,'{}_SEG.npy'.format(row[1])), dx*s, RR, saveCSVFile)
                    # else:
                    #     print('{} done, skipping'.format(magFile))

if __name__ == "__main__":
    csvFile = sys.argv[1]
    saveCSVFile = sys.argv[2]
    cutDataForStudy(csvFile,saveCSVFile)
