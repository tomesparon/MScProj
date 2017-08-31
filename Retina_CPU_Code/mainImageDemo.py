# -*- coding: utf-8 -*-
"""

@author: Tom

Demoing colour opponency and DoG on still images
"""

import cv2
import numpy as np
import scipy
import os
import retina
import cortex
import rgc
import matplotlib.pyplot as plt

mat_data = os.getcwd() + os.sep + 'ozv1retinas'
coeff = [0, 0, 0, 0]
loc = [0, 0, 0, 0]
coeff[0] = scipy.io.loadmat(mat_data + '\coeff8k_o.mat')['coeff8k']
coeff[1] = scipy.io.loadmat(mat_data + '\coeff4k_o.mat')['coeff4k']
coeff[2] = scipy.io.loadmat(mat_data + '\coeff1k_o.mat')['coeff1k']
coeff[3] = scipy.io.loadmat(mat_data + '\coeff256_o.mat')['tess256']
loc[0] = scipy.io.loadmat(mat_data + '\loc8k_o.mat')['loc8k']
loc[1] = scipy.io.loadmat(mat_data + '\loc4k_o.mat')['loc4k']
loc[2] = scipy.io.loadmat(mat_data + '\loc1k_o.mat')['loc1k']
loc[3] = scipy.io.loadmat(mat_data + '\loc256_o.mat')['tess256']

dcoeff = [0, 0, 0, 0]
dloc = [0, 0, 0, 0]
dcoeff[0] = scipy.io.loadmat(mat_data + '\coeff8k_od.mat')['coeff8k']
dcoeff[1] = scipy.io.loadmat(mat_data + '\coeff4k_od.mat')['coeff4k']
dcoeff[2] = scipy.io.loadmat(mat_data + '\coeff1k_od.mat')['coeff1k']
dcoeff[3] = scipy.io.loadmat(mat_data + '\coeff256_od.mat')['tess256']
dloc[0] = scipy.io.loadmat(mat_data + '\loc8k_od.mat')['loc8k']
dloc[1] = scipy.io.loadmat(mat_data + '\loc4k_od.mat')['loc4k']
dloc[2] = scipy.io.loadmat(mat_data + '\loc1k_od.mat')['loc1k']
dloc[3] = scipy.io.loadmat(mat_data + '\loc256_od.mat')['tess256']


i = 0

showInverse = True
showCortex = True

font = cv2.FONT_HERSHEY_PLAIN
types = ["RG","GR","RGinv","GRinv","BY","YB","BYinv","YBinv"]

global  L, R, L_loc, R_loc, G, cort_size
L, R = cortex.LRsplit(loc[i])
L_loc, R_loc = cortex.cort_map(L, R)
L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

stdimg_dir = os.getcwd() + os.sep + 'testimage\\'
print "Using " + os.listdir(stdimg_dir)[0]
name = os.listdir(stdimg_dir)[0]
img = cv2.imread(stdimg_dir+name, )
x, y = img.shape[1]/2, img.shape[0]/2
xx, yy = 1, img.shape[0]/10
imgsize = img.shape


GI = retina.gauss_norm_img(x, y, dcoeff[i], dloc[i], imsize=imgsize,rgb=True)



def showNonOpponency(C,theta):
    GI = retina.gauss_norm_img(x, y, dcoeff[i], dloc[i], imsize=imgsize,rgb=False)
    S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True)

    ncentreV,nsurrV = rgc.nonopponency(C,S,theta)
    print nsurrV.shape
    ninverse = retina.inverse(ncentreV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    ninv_crop = retina.crop(ninverse,x,y,dloc[i])
    ninverse2 = retina.inverse(nsurrV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    ninv_crop2 = retina.crop(ninverse2,x,y,dloc[i])
    cv2.putText(ninv_crop,"R+G + ",(xx,yy), font, 1,(255,255,255),2)
    cv2.putText(ninv_crop2,"R+G - ",(xx,yy), font, 1,(255,255,255),2)
    merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
    

    lposnon, rposnon = cortex.cort_img(ncentreV, L, L_loc, R, R_loc, cort_size, G)
    lnegnon, rnegnon = cortex.cort_img(nsurrV, L, L_loc, R, R_loc, cort_size, G)
    pos_cort_img = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)
    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    return merged, mergecort
        

def showBPImg(pV,nV):
    inv_crop = np.empty(8, dtype=object)
    inv_crop2 = np.empty(8, dtype=object)
    for t in range(8):
        inverse = retina.inverse(pV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
        inv_crop[t] = retina.crop(inverse,x,y,dloc[i])

        inverse2 = retina.inverse(nV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
        inv_crop2[t] = retina.crop(inverse2,x,y,dloc[i])

        cv2.putText(inv_crop[t],types[t] + " + ",(xx,yy), font, 1,(0,255,255),2)
        cv2.putText(inv_crop2[t],types[t] + " - ",(xx,yy), font, 1,(0,255,255),2)

    posRG = np.vstack((inv_crop[:4]))
    negRG = np.vstack((inv_crop2[:4]))
    posYB = np.vstack((inv_crop[4:]))
    negYB = np.vstack((inv_crop2[4:]))
    merge = np.concatenate((posRG,negRG,posYB,negYB),axis=1)
    return merge


def showCortexImg(pV,nV):
    pos_cort_img = np.empty(8, dtype=object)
    neg_cort_img = np.empty(8, dtype=object)
    for t in range(8):
        lpos, rpos = cortex.cort_img(pV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
        lneg, rneg = cortex.cort_img(nV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
        pos_cort_img[t] = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
        neg_cort_img[t] = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)

    posRGcort = np.vstack((pos_cort_img[:4]))
    negRGcort = np.vstack((neg_cort_img[:4]))
    posYBcort = np.vstack((pos_cort_img[4:]))
    negYBcort = np.vstack((neg_cort_img[4:]))
    mergecort = np.concatenate((posRGcort,negRGcort,posYBcort,negYBcort),axis=1)
    return mergecort


def imagetest(thetainput,doubleopponencyinput):
    theta = thetainput
    rgcMode = doubleopponencyinput


    C = retina.sample(img,x,y,coeff[i],loc[i],rgb=True) # CENTRE
    S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True) # SURROUND
    
    if rgcMode == 0:
    	pV,nV = rgc.opponency(C,S,theta)
    else:
    	pV,nV = rgc.doubleopponency(C,S,theta)
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.imshow("Input", img)
    rIntensity,cIntensity = showNonOpponency(C,theta)
    cv2.namedWindow("Intensity Responses", cv2.WINDOW_NORMAL)
    cv2.imshow("Intensity Responses", rIntensity)
    cv2.namedWindow("Intensity Responses Cortex", cv2.WINDOW_NORMAL)
    cv2.imshow("Intensity Responses Cortex", cIntensity)
    cv2.waitKey(0)
    #Generate backprojected images
    if showInverse:
        rOpponent = showBPImg(pV,nV)
        cv2.namedWindow("Backprojected Opponent Cells Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Backprojected Opponent Cells Output", rOpponent)
        cv2.waitKey(0)
    # Cortex
    if showCortex:
        cOpponent = showCortexImg(pV,nV)
        cv2.namedWindow("Cortex Opponent Cells Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Cortex Opponent Cells Output", cOpponent)
        cv2.waitKey(0)


def imagetestplt(thetainput,doubleopponencyinput):


    theta = thetainput
    rgcMode = doubleopponencyinput


    C = retina.sample(img,x,y,coeff[i],loc[i],rgb=True) # CENTRE
    S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True) # SURROUND
    
    if rgcMode == 0:
        pV,nV = rgc.opponency(C,S,theta)
    else:
        pV,nV = rgc.doubleopponency(C,S,theta)
    # cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    # cv2.imshow("Input", img)
    rIntensity,cIntensity = showNonOpponency(C,theta)

    plt.subplot(3,1,1), plt.imshow(cv2.cvtColor(rIntensity, cv2.COLOR_BGR2RGB)), plt.title('Backprojected R+G Intensity Response')
    plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,2), plt.imshow(cv2.cvtColor(cIntensity, cv2.COLOR_BGR2RGB)), plt.title('Cortical R+G Intensity Response')
    plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,3), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original test image')
    plt.xticks([]), plt.yticks([])
    thetastring = "%.2f" % theta
    plt.suptitle('Rectified DoG Intensity Images. Threshold:' + thetastring, fontsize=16)
    plt.show()

    #Generate backprojected images
    if showInverse:
        rOpponent = showBPImg(pV,nV)
        plt.imshow(cv2.cvtColor(rOpponent, cv2.COLOR_BGR2RGB)), plt.title('Backprojected Opponent Cells Output')
        plt.xticks([]), plt.yticks([])
        plt.show()
    # Cortex
    if showCortex:
        cOpponent = showCortexImg(pV,nV)
        plt.imshow(cv2.cvtColor(cOpponent, cv2.COLOR_BGR2RGB)), plt.title('Cortex Opponent Cells Output')
        plt.xticks([]), plt.yticks([])
        plt.show()





def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


print "Welcome to the Still Image Colour Opponency creation utility.\n\n"

while True:
    try:
        arg1 = raw_input("Select theta between 0 and 1:").lower()
        arg1 = float(arg1)
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    if arg1 < 0 or arg1 >1:
        print("Sorry, your response must be 0-1")
        continue
    else:
        break

arg2 = raw_input("Set 'true' for doubleopponency")
arg2 = str2bool(arg2)
print arg1
print arg2

arg3 = raw_input("Set 'true' for MatPlotLib Plot")
arg3 = str2bool(arg3)

if arg3 == 0:
    imagetest(arg1, arg2)
else:
    imagetestplt(arg1, arg2)
