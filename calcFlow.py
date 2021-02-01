import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil
import sys
import os
import csv

def calcFlow(filePath, width, RR, saveCSVFile):
    # print('Input file path = {}'.format(filePath))
    if not os.path.exists(filePath):
        filePath = '\\\\MPUFS7\\data_mrcv\\45_DATA_HUMANS\\CHEST\\STUDIES\\2017_Eldridge_NLP_MET\\002\\Normoxia\\CS_WAVELET_20_FRAMES_LOWRES\\dat\\CORRECTED\\Aorta_SEG.npy'
        print('File not found, using default instead.')
    print('File path = {}'.format(filePath))
    # print('Grid width in mm = {}'.format(width))

    seg = np.load(filePath)
    velNormal = np.load(filePath.replace('_SEG','_velNormal'))
    dx = width/np.size(velNormal,axis=0)
    nT = np.size(velNormal, axis=2)
    segT = []
    backwardMask = velNormal<0
    flow = np.zeros(nT)
    revFlow = np.zeros(nT)

    for tt in range(nT):
        segT.append(seg)
        revFlow[tt] = np.sum(-velNormal[:,:,tt]*backwardMask[:,:,tt]*seg)*dx*dx/1000.0 # in ml/second
        flow[tt] = np.sum(velNormal[:,:,tt]*seg)*dx*dx/1000.0 # in ml/second

    segT = np.moveaxis(segT,0,-1)
    maskedVel = velNormal*segT
    path = os.path.dirname(filePath)
    vessel = os.path.basename(filePath).replace('_SEG.npy','')
    meanFlow = np.mean(flow*60.0/1000.0) # in L/min
    flowPerCycle = np.mean(flow)*RR/1000.0
    regurgitantFraction = np.sum(revFlow)/np.sum(flow)
    maxVelocity = np.amax(maskedVel)
    meanVelocity = np.mean(maskedVel)
    pIndex = (max(flow)-min(flow))/np.mean(flow)
    rIndex = (max(flow)-min(flow))/max(flow)
    data = [path, vessel, meanFlow, flowPerCycle, regurgitantFraction, maxVelocity, meanVelocity, pIndex, rIndex]

    # plt.figure()
    # plt.plot(flow)

    print('Mean Flow = {:.2f} L/min'.format(meanFlow))
    # plt.show()

    for flowVal in flow:
        data.append(flowVal)

    if not os.path.exists(saveCSVFile):
        with open(saveCSVFile, mode='w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Path', 'Vessel', 'Mean Flow', 'Flow per Cycle', 'Regurgitant Fraction', 'Max Velocity', 'Mean Velocity', 'PI', 'RI', 'Flow Waveform'])
    else:
        with open(saveCSVFile, mode='a') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(data)
    # print('Saved {}'.format(filePath))

if __name__ == "__main__":
    filePath = sys.argv[1]
    width = float(sys.argv[2])
    RR = float(sys.argv[3])
    saveCSVFile = sys.argv[4]
    calcFlow(filePath, width, RR, saveCSVFile)
