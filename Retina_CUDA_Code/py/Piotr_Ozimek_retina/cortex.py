# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 22:40:44 2017

Cortical image functions.
@author: Piotr Ozimek
"""
import numpy as np
import math
import scipy.io
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import retina
import matplotlib.image as im
import scipy.ndimage
import cv2 

test = False

"""Split the receptive fields into two halves, left and right"""
        #TODO + 2 overlapping halves, ideally overlap is in its own set
def LRsplit(rf_loc):
    left = []
    right = []
    #alpha_loc = np.zeros(rf_loc.k_width/2,2) size for alpha strip????
    for i in range(len(rf_loc)):
        rf_loc[i,2] = i
        if rf_loc[i,0] < 0:
            left.append(rf_loc[i,:])
        else:
            right.append(rf_loc[i,:])

    L = np.zeros(shape=(3,len(left))) #4057
    R = np.zeros(shape=(3,len(right))) #4135  (+78, its fine)
    
    L = np.array(left)
    R = np.array(right)
    
    # [x,y,V_index]
    return L,R

def cort_map(L,R, alpha=15):
    #compute cortical coordinates
    L_r = np.sqrt((L[:,0]-alpha)**2 + L[:,1]**2)
    R_r = np.sqrt((R[:,0]+alpha)**2 + R[:,1]**2)
    L_theta = np.arctan2(L[:,1],L[:,0]-alpha) 
    L_theta = L_theta-np.sign(L_theta)*math.pi  #** shift theta by pi
    R_theta = np.arctan2(R[:,1],R[:,0]+alpha)
    L_loc = np.array([L_theta,L_r]).transpose()
    R_loc = np.array([R_theta,R_r]).transpose()    
    
    ##equate mean distances along x and y axis
    #x (theta)
    L_theta = np.zeros_like(L_loc)
    R_theta = np.zeros_like(R_loc)
    L_theta[:,0] = L_loc[:,0]
    R_theta[:,0] = R_loc[:,0]
    L_xd = np.mean(distance.cdist(L_theta,L_theta))
    R_xd = np.mean(distance.cdist(R_theta,R_theta))
    xd = (L_xd+R_xd)/2
    
    #y (r)
    L_r = np.zeros_like(L_loc)
    R_r = np.zeros_like(R_loc)
    L_r[:,1] = L_loc[:,1]
    R_r[:,1] = R_loc[:,1]
    L_yd = np.mean(distance.cdist(L_r,L_r))
    R_yd = np.mean(distance.cdist(R_r,R_r))
    yd = (L_yd+R_yd)/2
    
    #scale theta (x)
    L_loc[:,0] *= yd/xd
    R_loc[:,0] *= yd/xd
    
    return L_loc, R_loc

#TODO: shrinking factor automatically selected to make dist_5=0.75 ??
def cort_prepare(L_loc, R_loc, shrink=0.5, k_width=7, sigma = 0.8):
    G=gauss100(k_width,sigma)
    
    #bring min(x) to 0
    L_loc[:,0] -= np.min(L_loc[:,0])
    R_loc[:,0] -= np.min(R_loc[:,0])
    #flip y and bring min(y) to 0
    L_loc[:,1] = -L_loc[:,1]
    R_loc[:,1] = -R_loc[:,1]
    L_loc[:,1] -= np.min(L_loc[:,1])
    R_loc[:,1] -= np.min(R_loc[:,1])
    
    #k_width more pixels of space from all sides for kernels to fit
    L_loc += k_width
    R_loc += k_width
    cort_y = np.max(L_loc[:,1]) + k_width
    cort_x = np.max(L_loc[:,0]) + k_width
    

    #shrinking
    cort_size = (int(cort_y*shrink),int(cort_x*shrink))
    L_loc *= shrink
    R_loc *= shrink
    
    return L_loc, R_loc, G, cort_size

def cort_img(V, L, L_loc, R, R_loc, cort_size, G, k_width=7, sigma = 0.8):
    rgb = len(V.shape) == 2
    if rgb: cort_size = (cort_size[0], cort_size[1], 3)
    
    ##Project cortical images
    L_img = np.zeros(cort_size)
    R_img = np.zeros(cort_size)
    L_gimg = np.zeros(cort_size[:2]) #normalization images dont need depth
    R_gimg = np.zeros(cort_size[:2])
    
    #L
    for p in range(len(L_loc)):
        #coords into img array
        x = int(round(L_loc[p,0]))
        y = int(round(L_loc[p,1]))
        #coords of kernel in img array
        y1 = y - k_width/2 if (y - k_width/2) > 0 else 0
        y2 = y + k_width/2 + 1 
        x1 = x - k_width/2 if (x - k_width/2) > 0 else 0
        x2 = x + k_width/2 + 1 
        #coords into the 10x10 gaussian filters array
        dx = int(10*(np.round(L_loc[p,0], decimals=1) - round(L_loc[p,0])))
        dy = int(10*(np.round(L_loc[p,1], decimals=1) - round(L_loc[p,1])))
        #in case of big kernels, clipping kernels at img edges
        gy1 = 0 if (y - k_width/2) >= 0 else -(y - k_width/2)
        gy2 = k_width if y2 <= cort_size[0] else k_width-(y2-cort_size[0])
        gx1 = 0 if (x - k_width/2) >= 0 else -(x - k_width/2)
        gx2 = k_width if x2 <= cort_size[1] else k_width-(x2-cort_size[1])
        
        g = G[dx,dy][gy1:gy2,gx1:gx2]
        
        if rgb: add = np.dstack((g, g, g)) * V[int(L[p,2])]
        else: add = g * V[int(L[p,2])]
        
        L_img[y1:y2,x1:x2] += add
        L_gimg[y1:y2,x1:x2] += g
     
    if rgb: L_gimg = np.dstack((L_gimg, L_gimg, L_gimg))
    left = np.uint8(np.divide(L_img, L_gimg))
    
    #R
    for p in range(len(R_loc)):
        #coords into img array
        x = int(round(R_loc[p,0]))
        y = int(round(R_loc[p,1]))
        #coords of kernel in img array
        y1 = y - k_width/2 if (y - k_width/2) > 0 else 0
        y2 = y + k_width/2 + 1 
        x1 = x - k_width/2 if (x - k_width/2) > 0 else 0
        x2 = x + k_width/2 + 1 
        #coords into the 10x10 gaussian filters array
        dx = int(10*(np.round(R_loc[p,0], decimals=1) - round(R_loc[p,0])))
        dy = int(10*(np.round(R_loc[p,1], decimals=1) - round(R_loc[p,1])))
        #in case of big kernels, clipping kernels at img edges
        gy1 = 0 if (y - k_width/2) >= 0 else -(y - k_width/2)
        gy2 = k_width if y2 <= cort_size[0] else k_width-(y2-cort_size[0])
        gx1 = 0 if (x - k_width/2) >= 0 else -(x - k_width/2)
        gx2 = k_width if x2 <= cort_size[1] else k_width-(x2-cort_size[1])
    
        g = G[dx,dy][gy1:gy2,gx1:gx2]
        
        if rgb: add = np.dstack((g, g, g)) * V[int(R[p,2])]
        else: add = g * V[int(R[p,2])]
        
        R_img[y1:y2,x1:x2] += add
        R_gimg[y1:y2,x1:x2] += g
    
    if rgb: R_gimg = np.dstack((R_gimg, R_gimg, R_gimg))
    right = np.uint8(np.divide(R_img,R_gimg))
    
    return left, right

def show_cortex(left,right):
#    rgb = len(left.shape) == 3
    print 'original scale' 
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print 'Larger...'
    LRfig = plt.figure(figsize=(10,5))
    LRgrid = ImageGrid(LRfig, 111, nrows_ncols=(1,2))
    LRgrid[0].imshow(left, interpolation='none')
    LRgrid[1].imshow(right, interpolation='none')
    plt.show()

"""10x10 Gaussian Functions"""
#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel(width,loc,sigma):
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    

    #subpixel accurate coords of gaussian centre
    dx = width/2 + np.round(loc[0],decimals=1) - int(loc[0])
    dy = width/2 + np.round(loc[1],decimals=1) - int(loc[1])    
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,dx-x,dy-y)
    
    return k

#10x10 kernels
def gauss100(width,sigma):
    data = np.zeros((10,10,width,width))
    offsets = np.linspace(0,0.9,10)
    for x in range(10):
        for y in range(10):
            data[x,y] = gausskernel(width,offsets[[x,y]],sigma)
    
    return data

"""Receptive Field Centres Stats"""
def rf_stats(rf_locations):
    dist = distance.cdist(rf_locations, rf_locations)
    
    #find the 2 points closest to each other
    nz = np.nonzero(dist)
    argmin = (nz[0][np.argmin(dist[nz])], nz[1][np.argmin(dist[nz])])

    L_min = dist[argmin]
    print "indices of closest pairs: ", argmin
    print "Distances of closest pairs: ", L_min
    print "r values for closest pair: ", rf_locations[argmin[0],1]

    dist.sort()
    
    #5 closest neighbours
    neigh = dist[:,1:6].flatten()
    avg = np.mean(neigh)
    
    plt.figure(figsize=(8,5))
    
    plt.title("Distances to 5 closest neighbours")
    plt.ylabel("Projection centres")
    plt.xlim(0,6)
    n, bins, patches = plt.hist(neigh, 50)
    
    plt.xlim(0,6)
    plt.show()
    del(dist)
    print "Average: "+str(avg)

"""tests"""
if (test):
    mat_data = 'C:\Users\walsie\RETINA\mat data'
    img_dir = 'C:\Users\walsie\Pictures\standard_test_images'
    rf_coeff = scipy.io.loadmat(mat_data+'\coefficients.mat')['M']
    #V = scipy.io.loadmat(mat_data+'\V.mat')['V'][0]
    rf_loc = scipy.io.loadmat(mat_data+'\locations.mat')['ind']
    rf_loc[:,:2] /= 2
    
    img = im.imread(img_dir+"\lena_gray_256.tif")#[:,:,0]
    #img = scipy.ndimage.interpolation.zoom(img,2)
    fixation_x = img.shape[1]/2
    fixation_y = img.shape[0]/2
    V = retina.retina_sample(img,fixation_x,fixation_y,rf_coeff,rf_loc)
    
    L, R = LRsplit(rf_loc)
    L_loc, R_loc = cort_map(L, R)
    L_loc, R_loc, G, cort_size = cort_prepare(L_loc, R_loc,)
    l, r = cort_img(V, L, L_loc, R, R_loc, cort_size, G)
    show_cortex(l,r)