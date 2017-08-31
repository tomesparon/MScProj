# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:33:43 2016

@author: Piotr Ozimek

Retinal functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

"""Pad an image with 0s from all sides for ez sampling"""
def pad(img, padding, nans=False, rgb=False):
    size = (img.shape[0] + 2*padding, img.shape[1] + 2*padding)
    if rgb:
        s = list(size)
        s.append(3L)
        size = tuple(s)
    out = np.zeros(size, dtype = img.dtype)
    if nans:
        out = np.full(size,np.nan)
        
    out[padding:-padding, padding:-padding] = img
    return out
    

"""Sample an image given a retina"""
def sample(img, x, y, rf_coeff, rf_loc, rgb=False):    
    p = rf_loc[:,:2].max() + rf_loc[:,-1].max() #padding
    p = int(p) + 20
    pic = pad(img, p, True, rgb)
    
    s = len(rf_loc)
    X = rf_loc[:,0] + x + p
    Y = rf_loc[:,1] + y + p
    
    if rgb:
        V = np.zeros((s,3))
    else:
        V = np.zeros((s))
    
    for i in range(0,s):
        w = rf_loc[i,6]
        y1 = int(Y[i] - w/2+0.5)
        y2 = int(Y[i] + w/2+0.5)
        x1 = int(X[i] - w/2+0.5)
        x2 = int(X[i] + w/2+0.5)
        extract = pic[y1:y2,x1:x2]
        
        c = rf_coeff[0, i]
        if rgb:
            kernel = np.zeros((c.shape[0], c.shape[1], 3))
            kernel[:,:,0] = c
            kernel[:,:,1] = c
            kernel[:,:,2] = c
        else: kernel = c        
        
        m = np.where(np.isnan(extract), 0, 1.0) #mask
#        if m.shape != kernel.shape: print y1, y2, x1, x2, pic.shape, p, x, y #debug
        
        if rgb: f = 1.0/np.sum(m*kernel, axis = (0,1))
        else: f = 1.0/np.sum(m*kernel)
        
        extract = np.nan_to_num(extract)
        if rgb: V[i] = np.sum(extract*kernel, axis=(0,1)) * f
        else: V[i] = np.sum(extract*kernel) * f
        
    return V

#########

"""pre-generate gaussian normalization image for faster inversion"""
def gauss_norm_img(x, y, rf_coeff, rf_loc, imsize = (512,512), rgb=False):
    GI = np.zeros(imsize[:2])
    s = len(rf_loc) #num of receptive fields
    
    p = rf_loc[:,:2].max() + rf_loc[:,-1].max() #padding
    p = int(p) + 20
    GI = pad(GI, p, rgb)
        
    X = rf_loc[:,0] + x + p
    Y = rf_loc[:,1] + y + p
    
    for i in range(s-1,-1,-1):
        w = rf_loc[i,6]
        y1 = int(Y[i] - w/2+0.5)
        y2 = int(Y[i] + w/2+0.5)
        x1 = int(X[i] - w/2+0.5)
        x2 = int(X[i] + w/2+0.5)

        c = rf_coeff[0, i]    
        
        GI[y1:y2,x1:x2] += c
    
    GI = GI[p:-p,p:-p] #trim the padding
    
    return GI

def inverse(V, x, y, rf_coeff, rf_loc, GI, imsize = (512,512), rgb=False):
    if rgb: I1 = np.zeros((imsize[0], imsize[1], 3))
    else: I1 = np.zeros(imsize)
    p = rf_loc[:,:2].max() + rf_loc[:,-1].max() #padding
    p = int(p) + 20
    I1 = pad(I1, p, False, rgb)
    
    I = np.zeros(imsize)
    s = len(rf_loc) #num of receptive fields
    X = rf_loc[:,0] + x + p
    Y = rf_loc[:,1] + y + p
    
    for i in range(s-1,-1,-1):
        w = rf_loc[i,6]
        y1 = int(Y[i] - w/2+0.5)
        y2 = int(Y[i] + w/2+0.5)
        x1 = int(X[i] - w/2+0.5)
        x2 = int(X[i] + w/2+0.5)

        c = rf_coeff[0, i]
        if rgb:
            kernel = np.zeros((c.shape[0], c.shape[1], 3))
            kernel[:,:,0] = c
            kernel[:,:,1] = c
            kernel[:,:,2] = c
        else: kernel = c
        
        I1[y1:y2,x1:x2] += V[i] * kernel
    
    I1 = I1[p:-p, p:-p]
    if rgb: GI = np.dstack((GI,GI,GI))
    I = np.uint8(np.divide(I1,GI)) 
    return I

"""crop out the inversion lens from the large image"""
def crop(I, x, y, rf_loc):
    y = int(y)
    x = int(x)
    
    m = int(max(np.max(rf_loc[:,0]),np.max(rf_loc[:,1])) + 10)

    y1 = max(y-m,0)
    x1 = max(x-m,0)
    
    cropped = I[y1:y+m,x1:x+m]
    return cropped
    
"""project the inversion onto a statically sized image and center the fovea"""
def inverse_centre(V, rf_coeff, rf_loc, GI, rgb=False):
    w_max = np.max(rf_loc[:,6])
    m = int(2.0 * max(np.max(rf_loc[:,0]), np.max(rf_loc[:,1])) + w_max)
    
    if rgb: imsize = (m, m, 3)
    else: imsize = (m, m)
    
    I1 = np.zeros(imsize)
    I = np.zeros(imsize)
    
    s = len(rf_loc) #num of receptive fields
    X = rf_loc[:,0] + m/2.0
    Y = rf_loc[:,1] + m/2.0
    
    for i in range(s-1,-1,-1):
        w = rf_loc[i,6]
        y1 = int(Y[i] - w/2+0.5)
        y2 = int(Y[i] + w/2+0.5)
        x1 = int(X[i] - w/2+0.5)
        x2 = int(X[i] + w/2+0.5)

        c = rf_coeff[0, i]
        if rgb:
            kernel = np.zeros((c.shape[0], c.shape[1], 3))
            kernel[:,:,0] = c
            kernel[:,:,1] = c
            kernel[:,:,2] = c
        else: kernel = c
        
        I1[y1:y2,x1:x2] += V[i] * kernel

    if rgb: GI = np.dstack((GI,GI,GI))
    I = np.uint8(np.divide(I1,GI)) 
    return I


"""Pre-generate gaussian normalization image for centred inversion"""
def gauss_norm_centre(rf_coeff, rf_loc):
    w_max = np.max(rf_loc[:,6])
    m = int(2.0 * max(np.max(rf_loc[:,0]), np.max(rf_loc[:,1])) + w_max)
    imsize = (m, m)
    
    GI = np.zeros(imsize)
    s = len(rf_loc) #num of receptive fields
        
    X = rf_loc[:,0] + m/2.0
    Y = rf_loc[:,1] + m/2.0
    
    for i in range(s-1,-1,-1):
        w = rf_loc[i,6]
        y1 = int(Y[i] - w/2+0.5)
        y2 = int(Y[i] + w/2+0.5)
        x1 = int(X[i] - w/2+0.5)
        x2 = int(X[i] + w/2+0.5)

        c = rf_coeff[0, i]
        
        GI[y1:y2,x1:x2] += c

    return GI

def crop_fixation(im, x, y, rf_coeff, rf_loc, gnorm):
    w_max = np.max(rf_loc[:,6])
    m = int(2.0 * max(np.max(rf_loc[:,0]), np.max(rf_loc[:,1])) + w_max)
    r = int(m/2.0)
    
    ml = np.where(gnorm == 0, 0, 1.0)
    mask = np.dstack((ml,ml,ml))
    
    source = pad(im, r, rgb=True)
    X = x+r
    Y = y+r
    
    crop = np.zeros((m,m,3), dtype=im.dtype)
    crop[:] = mask * source[Y-r:Y+r, X-r:X+r]
    return crop
    
#########

"""Display statistics and dist_5 for the rfs"""
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