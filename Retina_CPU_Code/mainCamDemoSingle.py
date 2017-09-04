# -*- coding: utf-8 -*-
"""
Demoing colour opponency and DoG retinal cells with a webcam, displaying only one
colour opponent species at a given time.

@author: Tom Esparon

Attributes:
    coeff (list): Index of receptive field coefficients for a sharp retina
    dcoeff (list): Index of receptive field coefficients for larger 1.6x retina
    loc (list): Index of receptive field coefficients for a sharp retina
    dloc (list): Index of receptive field coefficients for larger 1.6x retina
    font (TYPE): Declared a font type for descriptions of cell types
    i (int): Integer index of the currently selected retina size
    mat_data (TYPE): Directory location of MAT files
    showCortex (bool): Boolean toggle to show/hide the cortical images
    showInverse (bool): Boolean toggle to show/hide the inverted images
    types (list): List of cell type abbreviated descriptions
    useVideo (bool): Boolean toggle for using a video stream instead

"""

import cv2
import numpy as np
import scipy
import os
import retina # Piotr Ozimek retina model
import cortex # Piotr Ozimek cortex model
import rgc # Retinal ganglion cell model

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

useVideo = False
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
    """Summary
    Small function that allows the trackbar to function
    """
    pass
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
# Construct trackbars onto input window
cv2.createTrackbar('theta','Input',0,100,nothing)
switch = 'Opponency\n'
cv2.createTrackbar(switch, 'Input',0,1,nothing)
species = "Species\n"
cv2.createTrackbar(species, 'Input',0,7,nothing)



def showNonOpponency(C,theta):
    """Summary
    This function encapsulates the routine to generate backprojected and cortical views for 
    the magnocellular pathway retinal ganglion cells
    Args:
        C (vector): The sharp retina is passed to the function
        theta (float): A threshold value is passed to the function
    
    Returns:
        merged: Return a merged image of the backprojected view as a numpy image array
        mergecort: Return a merged image of the cortical view as a numpy image array
    """
    # Sample using the other recepetive field, but with a temporally different image, lateimg
    S = retina.sample(lateimg,x,y,dcoeff[i],dloc[i],rgb=True)

    ncentreV,nsurrV = rgc.nonopponency(C,S,theta)
    ninverse = retina.inverse(ncentreV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=False)
    ninv_crop = retina.crop(ninverse,x,y,dloc[i])
    ninverse2 = retina.inverse(nsurrV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=False)
    ninv_crop2 = retina.crop(ninverse2,x,y,dloc[i])
    merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
    

    lposnon, rposnon = cortex.cort_img(ncentreV, L, L_loc, R, R_loc, cort_size, G)
    lnegnon, rnegnon = cortex.cort_img(nsurrV, L, L_loc, R, R_loc, cort_size, G)
    pos_cort_img = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)
    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    
    return merged,mergecort


def showBPImg(pV,nV,t):
    """Summary
    This function encapsulates the routine to generate rectified backprojected views of 
    one opponent retinal ganglion cell
    Args:
        pV (vector): Positive rectified imagevector
        nV (vector): Negative rectified imagevector
        t (int): Index position of opponent cell species
    Returns:
        merge: Return a merged image of all backprojected opponent cells as a numpy image array
    """
    # backprojection functions
    inverse = retina.inverse(pV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    inv_crop = retina.crop(inverse,x,y,dloc[i])

    inverse2 = retina.inverse(nV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    inv_crop2 = retina.crop(inverse2,x,y,dloc[i])
    # place descriptions
    cv2.putText(inv_crop,types[t] + " + ",(1,270), font, 1,(0,255,255),2)
    cv2.putText(inv_crop2,types[t] + " - ",(1,270), font, 1,(0,255,255),2)

    merge = np.concatenate((inv_crop,inv_crop2),axis=1)
    return merge


def showCortexImg(pV,nV,t):
    """Summary
    This function encapsulates the routine to generate rectified cortical views of 
    one opponent retinal ganglion cell type
    Args:
        pV (vector): Positive rectified imagevector
        nV (vector): Negative rectified imagevector
        t (int): Index position of opponent cell species
    
    Returns:
        mergecort: Return a merged image of all cortical opponent cells as a numpy image array
    """
    
    lpos, rpos = cortex.cort_img(pV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
    lneg, rneg = cortex.cort_img(nV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
    pos_cort_img = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)

    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    return mergecort


def prepRF():
    """Summary
    Helper function that is used to pre-generate the cortical map locations, 
    before the main loop
    """
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


# Start of main logic
prepRF()

# repeat for every new frame
while True:
    ret, img = cap.read()
    ret, lateimg = cap.read()
    if ret is True:
        # get image frame properties
        x = int(img.shape[1]/2)
        y = int(img.shape[0]/2)
        imgsize = (img.shape[0],img.shape[1])

        theta = cv2.getTrackbarPos('theta','Input') / 100.0
        rgcMode = cv2.getTrackbarPos(switch,'Input')
        # get index of the species type
        t = cv2.getTrackbarPos(species,'Input')
        # sample images
        C = retina.sample(img,x,y,coeff[i],loc[i],rgb=True) # CENTRE
        S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True) # SURROUND

        # generate rectified imagevectors based on the type of opponency
        if rgcMode == 0:
        	pV,nV = rgc.opponency(C,S,theta)
        else:
        	pV,nV = rgc.doubleopponency(C,S,theta)

        # Display functions
        cv2.imshow("Input", img)

        rIntensity,cIntensity = showNonOpponency(C,theta)
        cv2.imshow("Intensity Responses", rIntensity)
        cv2.namedWindow("Intensity Responses Cortex", cv2.WINDOW_NORMAL)
        cv2.imshow("Intensity Responses Cortex", cIntensity)
        #Generate backprojected images
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

#Ran if camera stops working
cv2.destroyAllWindows()
cap.release()
cv2.VideoCapture(camid).release()