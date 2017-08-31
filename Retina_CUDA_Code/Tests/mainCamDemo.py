#!/usr/bin/python

# -*- coding: utf-8 -*-
"""

@author: Tom

Demoing colour opponency and DoG
"""


import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)
import scipy
import cPickle as pickle
import time
import sys
sys.path.append('../py')
sys.path.append('../py/Piotr_Ozimek_retina')
import retina_cuda
import cortex_cuda
import retina
import cortex
import rgc
import os
import rgc
import itertools


coeff = [0,0,0,0]
loc = [0,0,0,0]

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
# theta = 0.0
font = cv2.FONT_HERSHEY_PLAIN
types = ["RG","GR","RGinv","GRinv","BY","YB","BYinv","YBinv"]

#### TRACKBAR
def nothing(x):
    pass
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.createTrackbar('theta','Input',0,100,nothing)
switch = 'Opponency\n'
cv2.createTrackbar(switch, 'Input',0,1,nothing)
motion = 'Temporal'
cv2.createTrackbar(motion, 'Input',0,1,nothing)
###

showInverse = True
showCortex = True

camid = 1
cap = cv2.VideoCapture(camid)
while not cap.isOpened():
    print 'retrying\n'
    cv2.VideoCapture(camid).release()
    cap = cv2.VideoCapture(camid)
    camid += 1

ret, img = cap.read()
ret0 = retina_cuda.create_retina(loc[0], coeff[0], img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
ret1 = retina_cuda.create_retina(dloc[0], dcoeff[0], img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
cort0 = cortex_cuda.create_cortex_from_fields(loc[0], rgb=True)
cort1 = cortex_cuda.create_cortex_from_fields(dloc[0],  rgb=True)
#retg0 = retina_cuda.create_retina(loc[0], coeff[0], (img.shape(0), img.shape(1)), (int(img.shape[1]/2), int(img.shape[0]/2)))
#retg1 = retina_cuda.create_retina(dloc[0], dcoeff[0], (img.shape(0), img.shape(1)), (int(img.shape[1]/2), int(img.shape[0]/2)))

while True:
    ret, img = cap.read()
    if ret is True:
        theta = cv2.getTrackbarPos('theta','Input') / 100.0
        rgcMode = cv2.getTrackbarPos(switch,'Input')
        motionMode = cv2.getTrackbarPos(motion,'Input')
        #if not colourmode: img = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
        
        if motionMode==1:
            ret, lateimg = cap.read()
            C = ret0.sample(img) # CENTRE
            S = ret0.sample(lateimg) # SURROUND
        else:
            C = ret0.sample(img) # CENTRE
            S = ret1.sample(img) # SURROUND

        C = retina_cuda.convert_to_Piotr(C)
        S = retina_cuda.convert_to_Piotr(S)

        #if not doubleopponency:centre,surr = rgc.opponency(C,S,theta)
        #else:centre,surr = rgc.doubleopponency(C,S,theta)
        
        #ncentre,nsurr = rgc.nonopponency(C,S,theta)

        cv2.imshow("Input", img)
        if showInverse:
            imgsize = (img.shape[0],img.shape[1])
            theta = cv2.getTrackbarPos('theta','Input') / 100.0

            cv2.imshow("Input", img)
            
            ### SINGLE AND DOUBLE OPPONENCY ROUTINE BP ###
            if rgcMode == 0:
                pV, nV = rgc.opponency(C, S,theta)
            else:
                pV, nV = rgc.doubleopponency(C, S,theta)
            # print pV.shape
            
            inv_crop = np.empty(8, dtype=object)
            inv_crop2 = np.empty(8, dtype=object)
            for i in range(8):  
                inverse = ret0.inverse(retina_cuda.convert_from_Piotr(pV[:,i,:].astype(float)))
                inv_crop[i] = retina.crop(inverse,int(img.shape[1]/2), int(img.shape[0]/2),loc[0])
               
                inverse2 = ret1.inverse(retina_cuda.convert_from_Piotr(nV[:,i,:].astype(float)))
                inv_crop2[i] = retina.crop(inverse2,int(img.shape[1]/2), int(img.shape[0]/2),dloc[0])
                cv2.putText(inv_crop[i],types[i] + " + ",(1,270), font, 1,(0,255,255),2)
                cv2.putText(inv_crop2[i],types[i] + " - ",(1,270), font, 1,(0,255,255),2)
            posRG = np.vstack((inv_crop[:4]))
            negRG = np.vstack((inv_crop2[:4]))
            posYB = np.vstack((inv_crop[4:]))
            negYB = np.vstack((inv_crop2[4:]))
            merge = np.concatenate((posRG,negRG,posYB,negYB),axis=1)
            
            cv2.namedWindow("Backprojected Opponent Cells Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Backprojected Opponent Cells Output", merge)
            # cv2.namedWindow("justone", cv2.WINDOW_NORMAL)
            # cv2.imshow("justone",inv_crop[0])
            ########################################

            ### NONOPPONENT/INTENSITY ROUTINE BP ###
            #ifshowinverse
            ncentre,nsurr = rgc.nonopponency(C,S,theta)
            ninverse = ret0.inverse(retina_cuda.convert_from_Piotr(ncentre.astype(float)))
            ninv_crop = retina.crop(ninverse,int(img.shape[1]/2), int(img.shape[0]/2),loc[0])
           
            ninverse2 = ret1.inverse(retina_cuda.convert_from_Piotr(nsurr.astype(float)))
            ninv_crop2 = retina.crop(ninverse2,int(img.shape[1]/2), int(img.shape[0]/2),dloc[0])

            cv2.putText(ninv_crop,"R+G + ",(1,270), font, 1,(0,255,255),2)
            cv2.putText(ninv_crop2,"R+G - ",(1,270), font, 1,(0,255,255),2)

            merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
            cv2.namedWindow("Intensity Responses", cv2.WINDOW_NORMAL)
            cv2.imshow("Intensity Responses", merged)

            #ifshowcortex
            lposnon = cort0.cort_image_left(retina_cuda.convert_from_Piotr(ncentre.astype(float)))
            rposnon = cort0.cort_image_right(retina_cuda.convert_from_Piotr(ncentre.astype(float)))
            lnegnon = cort1.cort_image_left(retina_cuda.convert_from_Piotr(nsurr.astype(float)))
            rnegnon = cort1.cort_image_right(retina_cuda.convert_from_Piotr(nsurr.astype(float)))
            pos_cort_img_non = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
            neg_cort_img_non = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)

            mergedcortex = np.concatenate((pos_cort_img_non, neg_cort_img_non),axis=1)
            cv2.namedWindow("Intensity Responses Cortex", cv2.WINDOW_NORMAL)
            cv2.imshow("Intensity Responses Cortex", mergedcortex)
            ############################################

    # Cortex
        if showCortex:
            pos_cort_img = np.empty(8, dtype=object)
            neg_cort_img = np.empty(8, dtype=object)
            for i in range(8):
                # print len(retina_cuda.convert_from_Piotr(pV[:,i,:].astype(float)))
                # print pV[:,i,:].astype(float).dtype
                lpos = cort0.cort_image_left(retina_cuda.convert_from_Piotr(pV[:,i,:].astype(float)))
                rpos = cort0.cort_image_right(retina_cuda.convert_from_Piotr(pV[:,i,:].astype(float)))
                lneg = cort1.cort_image_left(retina_cuda.convert_from_Piotr(nV[:,i,:].astype(float)))
                rneg = cort1.cort_image_right(retina_cuda.convert_from_Piotr(nV[:,i,:].astype(float)))
                pos_cort_img[i] = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
                neg_cort_img[i] = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)
                

            posRGcort = np.vstack((pos_cort_img[:4]))
            negRGcort = np.vstack((neg_cort_img[:4]))
            posYBcort = np.vstack((pos_cort_img[4:]))
            negYBcort = np.vstack((neg_cort_img[4:]))
            mergecort = np.concatenate((posRGcort,negRGcort,posYBcort,negYBcort),axis=1)
            cv2.namedWindow("Cortex Opponent Cells Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Cortex Opponent Cells Output", mergecort)








        # Response to key input settings
        key = cv2.waitKey(10)
        # if key == 43: ##check for '+'' key on numpad
        #     if i != 0: 
        #         i -= 1
        #         prep()
        # elif key == 45: #check for '-'' key on numpad
        #     if i != 3: 
        #         i += 1
        #         prep()
        if key == 117: #check for 'u' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            cv2.destroyWindow("Intensity Responses")
            showCortex = not showCortex
        elif key == 105: #check for 'i' key
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