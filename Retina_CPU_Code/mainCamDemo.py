# -*- coding: utf-8 -*-
"""

@author: Tom

Demoing colour opponency and DoG
"""
from timeit import default_timer as timer
import cv2
import numpy as np
import retina
import cortex
import scipy
import os
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
showCortex = False
colourmode = True
motion = False
doubleopponency = False
font = cv2.FONT_HERSHEY_PLAIN
types = ["RG","GR","RGinv","GRinv","BY","YB","BYinv","YBinv"]


print " +  increase retina size\n -  decrease retina size"
print " a  cortex autoscaling\nesc exit\n i show inverted retinal image"


#### TRACKBAR
def nothing(x):
    pass
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.createTrackbar('theta','Input',0,100,nothing)

while not cap.isOpened():
    print 'retrying\n'
    cv2.VideoCapture(camid).release()
    cap = cv2.VideoCapture(camid)
    camid -= 1

def prep():
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

prep()


while True:


    ret, img = cap.read()
    if ret is True:
        x = int(img.shape[1]/2)
        y = int(img.shape[0]/2)
        imgsize = (img.shape[0],img.shape[1])
        theta = cv2.getTrackbarPos('theta','Input') / 100.0
        if motion:
            ret, lateimg = cap.read()
            if ret is True:
                C = retina.sample(img,x,y,coeff[i],loc[i],rgb=colourmode)
                S = retina.sample(lateimg,x,y,coeff[i],loc[i],rgb=colourmode)
        else:
            C = retina.sample(img,x,y,coeff[i],loc[i],rgb=colourmode) # CENTRE
            S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=colourmode) # SURROUND
        
        if not doubleopponency:centre,surr = rgc.opponency(C,S,theta)
        else:centre,surr = rgc.doubleopponency(C,S,theta)
        
        ncentre,nsurr = rgc.nonopponency(C,S,theta)
        
        cv2.imshow("Input", img)
        #Generate backprojected images
        if showInverse:
            ninverse = retina.inverse(ncentre,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=colourmode)
            ninv_crop = retina.crop(ninverse,x,y,dloc[i])

            ninverse2 = retina.inverse(nsurr,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=colourmode)
            ninv_crop2 = retina.crop(ninverse2,x,y,dloc[i])
            merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
            cv2.imshow("Intensity Responses", merged)

            inv_crop = np.empty(8, dtype=object)
            inv_crop2 = np.empty(8, dtype=object)
            start = timer() # each loop takes 0.2 secs
            for t in range(8):
                    inverse = retina.inverse(centre[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=colourmode)
                    inv_crop[t] = retina.crop(inverse,x,y,dloc[i])

                    inverse2 = retina.inverse(surr[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=colourmode)
                    inv_crop2[t] = retina.crop(inverse2,x,y,dloc[i])

                    cv2.putText(inv_crop[t],types[t] + " + ",(1,270), font, 1,(0,255,255),2)
                    cv2.putText(inv_crop2[t],types[t] + " - ",(1,270), font, 1,(0,255,255),2)

            posRG = np.vstack((inv_crop[:4]))
            negRG = np.vstack((inv_crop2[:4]))
            posYB = np.vstack((inv_crop[4:]))
            negYB = np.vstack((inv_crop2[4:]))
            merge = np.concatenate((posRG,negRG,posYB,negYB),axis=1)

            end = timer()
            print(end - start)
            cv2.namedWindow("Backprojected Opponent Cells Output", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Backprojected Opponent Cells Output", merge)
        # Cortex
        if showCortex:
            pos_cort_img = np.empty(8, dtype=object)
            neg_cort_img = np.empty(8, dtype=object)
            for t in range(8):
                lpos, rpos = cortex.cort_img(centre[:,t,:], L, L_loc, R, R_loc, cort_size, G)
                lneg, rneg = cortex.cort_img(surr[:,t,:], L, L_loc, R, R_loc, cort_size, G)
                pos_cort_img[t] = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
                neg_cort_img[t] = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)
                # cv2.putText(pos_cort_img[t],types[t] + " + ",(1,1), font, 1,(0,255,255),2)
                # cv2.putText(neg_cort_img[t],types[t] + " - ",(1,1), font, 1,(0,255,255),2)

            posRGcort = np.vstack((pos_cort_img[:4]))
            negRGcort = np.vstack((neg_cort_img[:4]))
            posYBcort = np.vstack((pos_cort_img[4:]))
            negYBcort = np.vstack((neg_cort_img[4:]))
            mergecort = np.concatenate((posRGcort,negRGcort,posYBcort,negYBcort),axis=1)
            cv2.namedWindow("Cortex Opponent Cells Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Cortex Opponent Cells Output", mergecort)
        # Response to key input settings
        key = cv2.waitKey(10)
        if key == 43: ##check for '+'' key on numpad
            if i != 0: 
                i -= 1
                prep()
        elif key == 45: #check for '-'' key on numpad
            if i != 3: 
                i += 1
                prep()
        elif key == 117: #check for 'u' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            cv2.destroyWindow("Intensity Responses")
            showCortex = not showCortex
        elif key == 105: #check for 'i' key
            cv2.destroyWindow("Cortex Opponent Cells Output")
            cv2.destroyWindow("Backprojected Opponent Cells Output")
            cv2.destroyWindow("Intensity Responses")
            showInverse = not showInverse
        elif key == 111: # check for 'o' key
            # cv2.destroyWindow("Backprojected Opponent Cells Output")
            doubleopponency = not doubleopponency
            print "Switching double opponency to " + str(doubleopponency)
        elif key == 111: # check for 'o' key
            # cv2.destroyWindow("Backprojected Opponent Cells Output")
            motion = not motion
            print "Switching temporal response to " + str(motion)
        elif key == 27: #check for ESC key on numpad
            break

#Run this if cam stops working
cv2.destroyAllWindows()
cap.release()
cv2.VideoCapture(camid).release()