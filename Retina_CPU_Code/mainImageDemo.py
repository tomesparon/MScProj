# -*- coding: utf-8 -*-
"""

@author: Tom

Demoing colour opponency and DoG on still images

Attributes:
    coeff (list): Index of receptive field coefficients for a sharp retina
    dcoeff (list): Index of receptive field coefficients for larger 1.6x retina
    dloc (list): Index of receptive field coefficients for larger 1.6x retina
    font (TYPE): Declared a font type for descriptions of cell types
    i (int): Integer index of the currently selected retina size
    img (TYPE): Input Image array
    imgsize (TYPE): Dimensions of image
    loc (list): Index of receptive field coefficients for a sharp retina
    mat_data (TYPE): Directory location of MAT files
    name (TYPE): Name of image file
    showCortex (bool): Boolean toggle to show/hide the cortical images
    showInverse (bool): Boolean toggle to show/hide the inverted images
    stdimg_dir (TYPE): Directory of image file
    types (list): List of cell type abbreviated descriptions
"""

import cv2
import numpy as np
import scipy
import os
import retina # Piotr Ozimek retina model
import cortex # Piotr Ozimek cortex model
import rgc # Retinal ganglion cell model
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

# read in an image file
stdimg_dir = os.getcwd() + os.sep + 'testimage\\'
print "Using " + os.listdir(stdimg_dir)[0]
name = os.listdir(stdimg_dir)[0]
img = cv2.imread(stdimg_dir+name, )
x, y = img.shape[1]/2, img.shape[0]/2
xx, yy = 1, img.shape[0]/10
imgsize = img.shape

# generate gaussian normalised image
GI = retina.gauss_norm_img(x, y, dcoeff[i], dloc[i], imsize=imgsize,rgb=True)



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
    GI = retina.gauss_norm_img(x, y, dcoeff[i], dloc[i], imsize=imgsize,rgb=False)
     # Sample using the other recepetive field, note there is no temporal response with still images
    S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True)
    #backproject the imagevectors
    ncentreV,nsurrV = rgc.nonopponency(C,S,theta)
    ninverse = retina.inverse(ncentreV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    ninv_crop = retina.crop(ninverse,x,y,dloc[i])
    ninverse2 = retina.inverse(nsurrV,x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
    ninv_crop2 = retina.crop(ninverse2,x,y,dloc[i])
    # place descriptive text onto generated images
    cv2.putText(ninv_crop,"R+G + ",(xx,yy), font, 1,(255,255,255),2)
    cv2.putText(ninv_crop2,"R+G - ",(xx,yy), font, 1,(255,255,255),2)
    merged = np.concatenate((ninv_crop, ninv_crop2),axis=1)
    
    # create cortical maps of the imagevectors
    lposnon, rposnon = cortex.cort_img(ncentreV, L, L_loc, R, R_loc, cort_size, G)
    lnegnon, rnegnon = cortex.cort_img(nsurrV, L, L_loc, R, R_loc, cort_size, G)
    pos_cort_img = np.concatenate((np.rot90(lposnon),np.rot90(rposnon,k=3)),axis=1)
    neg_cort_img = np.concatenate((np.rot90(lnegnon),np.rot90(rnegnon,k=3)),axis=1)
    mergecort = np.concatenate((pos_cort_img,neg_cort_img),axis=1)
    return merged, mergecort
        

def showBPImg(pV,nV):
    """Summary
    This function encapsulates the routine to generate rectified backprojected views of 
    all opponent retinal ganglion cells
    Args:
        pV (vector): Positive rectified imagevector
        nV (vector): Negative rectified imagevector
    
    Returns:
        merge: Return a merged image of all backprojected opponent cells as a numpy image array
    """
    # object arrays of the positive and negative images
    inv_crop = np.empty(8, dtype=object)
    inv_crop2 = np.empty(8, dtype=object)
    for t in range(8):
        # backprojection functions
        inverse = retina.inverse(pV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
        inv_crop[t] = retina.crop(inverse,x,y,dloc[i])
        inverse2 = retina.inverse(nV[:,t,:],x,y,dcoeff[i],dloc[i], GI, imsize=imgsize,rgb=True)
        inv_crop2[t] = retina.crop(inverse2,x,y,dloc[i])
        # place descriptions
        cv2.putText(inv_crop[t],types[t] + " + ",(xx,yy), font, 1,(0,255,255),2)
        cv2.putText(inv_crop2[t],types[t] + " - ",(xx,yy), font, 1,(0,255,255),2)
    # stack all images into a grid
    posRG = np.vstack((inv_crop[:4]))
    negRG = np.vstack((inv_crop2[:4]))
    posYB = np.vstack((inv_crop[4:]))
    negYB = np.vstack((inv_crop2[4:]))
    merge = np.concatenate((posRG,negRG,posYB,negYB),axis=1)
    return merge


def showCortexImg(pV,nV):
    """Summary
    This function encapsulates the routine to generate rectified cortical views of 
    all opponent retinal ganglion cells
    Args:
        pV (vector): Positive rectified imagevector
        nV (vector): Negative rectified imagevector
    
    Returns:
        mergecort: Return a merged image of all cortical opponent cells as a numpy image array
    """
    # object arrays of the positive and negative images
    pos_cort_img = np.empty(8, dtype=object)
    neg_cort_img = np.empty(8, dtype=object)
    for t in range(8):
        # cortical mapping functions
        lpos, rpos = cortex.cort_img(pV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
        lneg, rneg = cortex.cort_img(nV[:,t,:], L, L_loc, R, R_loc, cort_size, G)
        pos_cort_img[t] = np.concatenate((np.rot90(lpos),np.rot90(rpos,k=3)),axis=1)
        neg_cort_img[t] = np.concatenate((np.rot90(lneg),np.rot90(rneg,k=3)),axis=1)
    # stack all images into a grid
    posRGcort = np.vstack((pos_cort_img[:4]))
    negRGcort = np.vstack((neg_cort_img[:4]))
    posYBcort = np.vstack((pos_cort_img[4:]))
    negYBcort = np.vstack((neg_cort_img[4:]))
    mergecort = np.concatenate((posRGcort,negRGcort,posYBcort,negYBcort),axis=1)
    return mergecort


def imagetest(thetainput,doubleopponencyinput):
    """Summary
    Display function that generates the final output images using opencv windows
    Args:
        thetainput (float): a threshold value for perception
        doubleopponencyinput (bool): A boolean toggle for changing the opponency mode
    """
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
    """Summary
    Display function that generates the final output images using MatplotLib windows
    Args:
        thetainput (float): a threshold value for perception
        doubleopponencyinput (bool): A boolean toggle for changing the opponency mode
    """
    theta = thetainput
    rgcMode = doubleopponencyinput


    C = retina.sample(img,x,y,coeff[i],loc[i],rgb=True) # CENTRE(sharp retina)
    S = retina.sample(img,x,y,dcoeff[i],dloc[i],rgb=True) # SURROUND(blurred retina)
    
    if rgcMode == 0:
        pV,nV = rgc.opponency(C,S,theta)
    else:
        pV,nV = rgc.doubleopponency(C,S,theta)

    rIntensity,cIntensity = showNonOpponency(C,theta)
    # Construct window plots
    plt.subplot(3,1,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original test image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,2), plt.imshow(cv2.cvtColor(rIntensity, cv2.COLOR_BGR2RGB)), plt.title('Backprojected R+G Intensity Response')
    plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,3), plt.imshow(cv2.cvtColor(cIntensity, cv2.COLOR_BGR2RGB)), plt.title('Cortical R+G Intensity Response')
    plt.xticks([]), plt.yticks([])
    # format float to string
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
  """Summary
  Conversion utility function to change a string to a boolean value
  Args:
      v (string): The string to be converted
  
  Returns:
      bool: If the string matches a list of excepted strings, return true
  """
  return v.lower() in ("yes", "true", "t", "1")

# Command Line Interface to request the type of image to be created
print "Welcome to the Still Image Colour Opponency creation utility.\n\n"

while True:
    try:
        # read in a value for theta
        arg1 = raw_input("Select theta between 0 and 1:").lower()
        arg1 = float(arg1)
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    #check if in correct range
    if arg1 < 0 or arg1 >1:
        print("Sorry, your response must be 0-1")
        continue
    else:
        break
# read in user input of opponency mode
arg2 = raw_input("Set 'true' for doubleopponency")
arg2 = str2bool(arg2)
print arg1
print arg2
# user input of the generated window type
arg3 = raw_input("Set 'true' for MatPlotLib Plot")
arg3 = str2bool(arg3)

if arg3 == 0:
    imagetest(arg1, arg2)
else:
    imagetestplt(arg1, arg2)
