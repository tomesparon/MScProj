# -*- coding: utf-8 -*-
"""

@author: Tom

Demoing colour opponency and DoG
"""

import cv2
import numpy as np
import scipy
import os
import retina
import cortex
import rgc

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
camid = 1
cap = cv2.VideoCapture(camid)
showInverse = True
showCortex = True

font = cv2.FONT_HERSHEY_PLAIN
types = ["RG","GR","RGinv","GRinv","BY","YB","BYinv","YBinv"]

print "USER KEYBOARD CONTROLS"
print " + to increase retina size\n - to decrease retina size"
print "esc - exit\ni - Toggle inverted retinal images\nu - Toggle cortical images"


#### TRACKBAR
def nothing(x):
    pass
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.createTrackbar('theta','Input',0,100,nothing)
switch = 'Opponency\n'
cv2.createTrackbar(switch, 'Input',0,1,nothing)
species = "Species\n"
cv2.createTrackbar(species, 'Input',0,7,nothing)

while not cap.isOpened():
    print 'retrying\n'
    cv2.VideoCapture(camid).release()
    cap = cv2.VideoCapture(camid)
    camid -= 1

def showNonOpponency(C,theta):

        S = retina.sample(lateimg,x,y,dcoeff[i],dloc[i],rgb=True)

        ncentreV,nsurrV = rgc.nonopponency(C,S,theta)
        ninverse = retina.inverse(ncentreV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=False)
        ninv_crop = retina.crop(ninverse,x,y,dloc[i])
        ninverse2 = retina.inverse(nsurrV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=False)
        ninv_crop2 = retina.crop(ninverse2,x,y,dloc[i])
        merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
        cv2.imshow("Intensity Responses", merged)

        lposnon, rposnon = cortex.cort_img(ncentreV, L, L_loc, R, R_loc, cort_size, G)
        lnegnon, rnegnon = cortex.cort_img(nsurrV, L, L_loc, R, R_loc, cort_size, G)
        pos_cort_img = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
        neg_cort_img = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)
        mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
        cv2.namedWindow("Intensity Responses Cortex", cv2.WINDOW_NORMAL)
        cv2.imshow("Intensity Responses Cortex", mergecort)



def showBPImg(pV,nV,t):
    # t = 1
    inverse = retina.inverse(pV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    inv_crop = retina.crop(inverse,x,y,dloc[i])

    inverse2 = retina.inverse(nV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    inv_crop2 = retina.crop(inverse2,x,y,dloc[i])

    cv2.putText(inv_crop,types[t] + " + ",(1,270), font, 1,(0,255,255),2)
    cv2.putText(inv_crop2,types[t] + " - ",(1,270), font, 1,(0,255,255),2)

    merge = np.concatenate((inv_crop,inv_crop2),axis=1)
    cv2.namedWindow("Backprojected Opponent Cells Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Backprojected Opponent Cells Output", merge)

def showCortexImg(pV,nV,t):
    # t = 1
    lpos, rpos = cortex.cort_img(pV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
    lneg, rneg = cortex.cort_img(nV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
    pos_cort_img = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)

    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    cv2.namedWindow("Cortex Opponent Cells Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Cortex Opponent Cells Output", mergecort)

def prepRF():
    global  L, R, L_loc, R_loc, G, cort_size
    L, R = cortex.LRsplit(loc[i])
    L_loc, R_loc = cortex.cort_map(L, R)
    L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)
    
    ret, img = cap.read()
    x = int(img.shape[1]/2)
    y = int(img.shape[0]/2)
    imgsize = (img.shape[0],img.shape[1])
    global GI
    GI = retina.gauss_norm_img(x, y, dcoeff[i], dloc[i], imsize=imgsize,rgb=True)

prepRF()


while True:
    ret, img = cap.read()
    ret, lateimg = cap.read()
    if ret is True:
        x = int(img.shape[1]/2)
        y = int(img.shape[0]/2)
        imgsize = (img.shape[0],img.shape[1])
        theta = cv2.getTrackbarPos('theta','Input') / 100.0
        rgcMode = cv2.getTrackbarPos(switch,'Input')
        t = cv2.getTrackbarPos(species,'Input')
 
        C = retina.sample(img,x,y,coeff[i],loc[i],rgb=True) # CENTRE
        S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True) # SURROUND
        
        if rgcMode == 0:
        	pV,nV = rgc.opponency(C,S,theta)
        else:
        	pV,nV = rgc.doubleopponency(C,S,theta)

        cv2.imshow("Input", img)

        showNonOpponency(C,theta)
        #Generate backprojected images
        if showInverse:
            showBPImg(pV,nV,t)
        # Cortex
        if showCortex:
            showCortexImg(pV,nV,t)

        # Response to key input settings
        key = cv2.waitKey(10)
        if key == 43: ##check for '+'' key on numpad
            if i != 0: 
                i -= 1
                prepRF()
        elif key == 45: #check for '-'' key on numpad
            if i != 3: 
                i += 1
                prepRF()
        elif key == 117: #check for 'u' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            showCortex = not showCortex
        elif key == 105: #check for 'i' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            showInverse = not showInverse
        elif key == 27: #check for ESC key on numpad
            break

#Run this if cam stops working
cv2.destroyAllWindows()
cap.release()
cv2.VideoCapture(camid).release()