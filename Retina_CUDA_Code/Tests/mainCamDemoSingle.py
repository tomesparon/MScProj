#!/usr/bin/python

# -*- coding: utf-8 -*-
"""

@author: Tom Esparon

Demoing colour opponency and DoG with CUDA
"""

import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)
import scipy.io
import sys
sys.path.append('../py')
sys.path.append('../py/Piotr_Ozimek_retina')
import retina_cuda
import cortex_cuda
import retina

import rgc
import os




mat_data = os.getcwd() + os.sep + 'RetinasTom'
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

font = cv2.FONT_HERSHEY_PLAIN
types = ["RG","GR","RGinv","GRinv","BY","YB","BYinv","YBinv"]


showInverse = True
showCortex = True
useVideo = True
print "USER KEYBOARD CONTROLS"
print " + to increase retina size\n - to decrease retina size"
print "esc - exit\ni - Toggle inverted retinal images\nu - Toggle cortical images"

if useVideo:
    camid = os.getcwd() + os.sep + 'testvideo'+ os.sep +'vtest.webm'
    cap = cv2.VideoCapture(camid)
    camid = os.getcwd() + os.sep + 'testvideo'+ os.sep +'vtest.webm'
    cap = cv2.VideoCapture(camid)
else:
    camid = 1
    cap = cv2.VideoCapture(camid)
    while not cap.isOpened():
        print 'retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1

ret, img = cap.read()

#### TRACKBAR
def nothing(x):
    pass
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.createTrackbar('theta','Input',0,100,nothing)
switch = 'Opponency\n'
cv2.createTrackbar(switch, 'Input',0,1,nothing)
species = "Species\n"
cv2.createTrackbar(species, 'Input',0,7,nothing)


def showNonOpponency(C,theta):
    
    S = ret1.sample(lateimg) # SURROUND
    S = retina_cuda.convert_to_Piotr(S)
    #showinverse
    ncentreV,nsurrV = rgc.nonopponency(C,S,theta)
    ninverse = ret0.inverse(retina_cuda.convert_from_Piotr(ncentreV.astype(float)))
    ninv_crop = retina.crop(ninverse,int(img.shape[1]/2), int(img.shape[0]/2),loc[0])
   
    ninverse2 = ret1.inverse(retina_cuda.convert_from_Piotr(nsurrV.astype(float)))
    ninv_crop2 = retina.crop(ninverse2,int(img.shape[1]/2), int(img.shape[0]/2),dloc[0])

    cv2.putText(ninv_crop,"R+G + ",(1,270), font, 1,(0,255,255),2)
    cv2.putText(ninv_crop2,"R+G - ",(1,270), font, 1,(0,255,255),2)

    merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)


    #showcortex
    lposnon = cort0.cort_image_left(retina_cuda.convert_from_Piotr(ncentreV.astype(float)))
    rposnon = cort0.cort_image_right(retina_cuda.convert_from_Piotr(ncentreV.astype(float)))
    lnegnon = cort1.cort_image_left(retina_cuda.convert_from_Piotr(nsurrV.astype(float)))
    rnegnon = cort1.cort_image_right(retina_cuda.convert_from_Piotr(nsurrV.astype(float)))
    pos_cort_img_non = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
    neg_cort_img_non = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)

    mergedcortex = np.concatenate((pos_cort_img_non, neg_cort_img_non),axis=1)
    return merged,mergedcortex


def showBPImg(pV,nV,t):
    inverse = ret0.inverse(retina_cuda.convert_from_Piotr(pV[:,t,:].astype(float)))
    inv_crop = retina.crop(inverse,int(img.shape[1]/2), int(img.shape[0]/2),loc[0])
   
    inverse2 = ret1.inverse(retina_cuda.convert_from_Piotr(nV[:,t,:].astype(float)))
    inv_crop2 = retina.crop(inverse2,int(img.shape[1]/2), int(img.shape[0]/2),dloc[0])
    cv2.putText(inv_crop,types[t] + " + ",(1,270), font, 1,(0,255,255),2)
    cv2.putText(inv_crop2,types[t] + " - ",(1,270), font, 1,(0,255,255),2)

    merge = np.concatenate((inv_crop,inv_crop2),axis=1)
    return merge



def showCortexImg(pV,nV,t):
    lpos = cort0.cort_image_left(retina_cuda.convert_from_Piotr(pV[:,t,:].astype(float)))
    rpos = cort0.cort_image_right(retina_cuda.convert_from_Piotr(pV[:,t,:].astype(float)))
    lneg = cort1.cort_image_left(retina_cuda.convert_from_Piotr(nV[:,t,:].astype(float)))
    rneg = cort1.cort_image_right(retina_cuda.convert_from_Piotr(nV[:,t,:].astype(float)))
    pos_cort_img = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)
        
    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    return mergecort




def prepRF(p):
    ret0 = retina_cuda.create_retina(loc[p], coeff[p], img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    ret1 = retina_cuda.create_retina(dloc[p], dcoeff[p], img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cort0 = cortex_cuda.create_cortex_from_fields(loc[p], rgb=True)
    cort1 = cortex_cuda.create_cortex_from_fields(dloc[p],  rgb=True)
    return ret0,ret1,cort0,cort1

ret0,ret1,cort0,cort1 = prepRF(0)
p = 0
while True:
    ret, img = cap.read()
    ret, lateimg = cap.read()
    if ret is True:
        theta = cv2.getTrackbarPos('theta','Input') / 100.0
        rgcMode = cv2.getTrackbarPos(switch,'Input')
        t = cv2.getTrackbarPos(species,'Input')

        C = ret0.sample(img) # CENTRE
        S = ret1.sample(img) # SURROUND

        C = retina_cuda.convert_to_Piotr(C)
        S = retina_cuda.convert_to_Piotr(S)

        cv2.imshow("Input", img)
        
        imgsize = (img.shape[0],img.shape[1])
        theta = cv2.getTrackbarPos('theta','Input') / 100.0

        cv2.imshow("Input", img)
        
        # switch opponency mode
        if rgcMode == 0:
            pV, nV = rgc.opponency(C, S,theta)
        else:
            pV, nV = rgc.doubleopponency(C, S,theta)

        rIntensity,cIntensity = showNonOpponency(C, theta)
        cv2.namedWindow("Intensity Responses", cv2.WINDOW_NORMAL)
        cv2.imshow("Intensity Responses", rIntensity)
        cv2.namedWindow("Intensity Responses Cortex", cv2.WINDOW_NORMAL)
        cv2.imshow("Intensity Responses Cortex", cIntensity)
        #backPorjected Opponency
        if showInverse:    
            rOpponent = showBPImg(pV,nV,t)
            cv2.namedWindow("Backprojected Opponent Cells Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Backprojected Opponent Cells Output", rOpponent)
        # Cortex
        if showCortex:
            cOpponent = showCortexImg(pV,nV,t)
            cv2.namedWindow("Cortex Opponent Cells Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Cortex Opponent Cells Output", cOpponent)
        # Response to key input settings
        key = cv2.waitKey(10)
        if key == 43: ##check for '+'' key on numpad
            if p != 0: 
                p -= 1
                ret0,ret1,cort0,cort1 = prepRF(p)
        elif key == 45: #check for '-'' key on numpad
            if p != 3: 
                p += 1
                ret0,ret1,cort0,cort1 = prepRF(p)
        if key == 105: #check for 'u' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            cv2.destroyWindow("Intensity Responses")
            showCortex = not showCortex
        elif key == 117: #check for 'i' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            cv2.destroyWindow("Intensity Responses")
            showInverse = not showInverse
        elif key == 27: #check for ESC key on numpad
            break

#Run this if cam stops working
cv2.destroyAllWindows()
cap.release()
cv2.VideoCapture(camid).release()