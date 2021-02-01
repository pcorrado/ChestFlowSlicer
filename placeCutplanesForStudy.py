import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
import sys
import os
import csv
import ChestFlowSlicer.ChestFlowSlicer as ChestFlowSlicer
import platform

def get_username():
    return os.getlogin()

def convert(str):
    return str if (platform.system()!='Windows') else str.replace('/data/','\\\\MPUFS7\\').replace('/','\\')
def convertHard(str):
    return str.replace('/data/','\\\\MPUFS7\\').replace('/','\\')

def placeCutplanesForStudy(scanListCSVFile,savePlaneCSVFile,eachUser=False):
    nameSet = set()
    if os.path.exists(savePlaneCSVFile):
        with open(savePlaneCSVFile,mode='r') as savePlaneCSVFileReader:
            reader = csv.reader(savePlaneCSVFileReader, delimiter=',')
            for row in reader:
                if row and ((not eachUser) or row[8]==get_username()) and row[1]=='LIPV':
                    nameSet.add(convertHard(row[0]))
    file=open(scanListCSVFile,mode='r')
    numFiles = len(file.readlines())
    file.close()
    counter=0
    with open(scanListCSVFile, mode='r') as csvReadFile:
        reader = csv.reader(csvReadFile, delimiter=',')
        for row in reader:
            counter+=1
            if row:
                cdFile = os.path.join(convert(row[0]),'CD.dat')
                if os.path.exists(cdFile):
                    print('CD file exists: {}'.format(cdFile))
                    thisSet = set()
                    thisSet.add(convertHard(row[0]))
                    if thisSet.issubset(nameSet):
                        print('Case already processed, skipping.')
                    else:
                        print('Case not processed, opening...')
                        vessels = []
                        with open(savePlaneCSVFile,mode='r') as savePlaneCSVFileReader:
                            reader2 = csv.reader(savePlaneCSVFileReader, delimiter=',')
                            for row2 in reader2:
                                if row2 and (convertHard(row[0])==convertHard(row2[0])) and ((not eachUser) or row2[8]==get_username()) :
                                    vessels.append((row2[1], float(row2[2]), float(row2[3]), float(row2[4]), float(row2[5]), float(row2[6]), float(row2[7])))
                        ChestFlowSlicer.ChestFlowSlicer(cdFile=cdFile, saveCSVFile=savePlaneCSVFile, mode='quit',extraString='Line {} of {}.'.format(counter-1,numFiles-1), vessels=vessels)
                else:
                    print('CD file does not exist: {}'.format(cdFile))
