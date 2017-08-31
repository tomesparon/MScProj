# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 11:33:45 2017

A Python version for Sumitha's receptive field generation code from the file
wilsonretina5fine_8192s.m
- Modified for DoG generation- Tom Esparon
@author: Piotr Ozimek
"""


import numpy as np
from scipy.spatial import distance

#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel(width,loc,sigma):
    w = float(width)
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    
    
    shift = (w-1)/2.0

    #subpixel accurate coords of gaussian centre
    dx = loc[0] - int(loc[0])
    dy = loc[1] - int(loc[1])    

    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,(x-shift)-dx,(y-shift)-dy)
    
    return k    

def tess_scale(tessellation, neighbourhood, min_rf):
    #compute node density metric dist_5
    d = distance.cdist(tessellation, tessellation)
    s = np.sort(d)
    dist_5 = np.mean(s[:,1:neighbourhood], 1)                   #r
    
    #compute dist_5 for most central 5 nodes
    fov_dist_5 = np.mean(dist_5[:5])                            #closest_rfs
    
    #set central_dist_5 to min_rf (impose min_rf parameter)
    scaled = tessellation*(1/fov_dist_5)*min_rf
    
    return scaled

def xy_sumitha(x,y,k_width):
    k_width = int(k_width) #this will change                 #d
    
    #if odd size mask -> round coordinates
    if k_width%2 != 0:
        cx = round(x)
        cy = round(y)
        
    #else if even size mask -> 1 decimal point coordinates (always .5)
    else:
        cx = round(x) + np.sign(x-round(x))*0.5
        cy = round(y) + np.sign(y-round(y))*0.5
    
    return cx, cy


"""A python version for Sumithas receptive field generation function. Output
should be nearly identical to the matlab version """
def rf_sumitha(tessellation, min_rf, sigma_ratio, sigma):
    rf_loc = np.zeros((len(tessellation),7))
    rf_coeff = np.ndarray((1,len(tessellation)),dtype='object')
    
    neighbourhood = 6 #5 closest nodes
    
    #compute node density metric dist_5
    d = distance.cdist(tessellation, tessellation)
    s = np.sort(d)
    dist_5 = np.mean(s[:,1:neighbourhood], 1)                   #r
    
    #set central_dist_5 to min_rf (impose min_rf parameter)
    fov_dist_5 = np.mean(dist_5[:5])                            #closest_rfs
    
    
    scaled = tessellation*(1/fov_dist_5)*min_rf
    rf_loc[:,:2] = scaled
    
    rf_loc[:,6] = np.ceil(sigma*(1/sigma_ratio)*(1/fov_dist_5)*min_rf*dist_5)
    
    for i in range(len(tessellation)):
        k_width = int(rf_loc[i,6])
        cx, cy = xy_sumitha(rf_loc[i,0], rf_loc[i,1], k_width)        
        
        rx = rf_loc[i][0] - cx
        ry = rf_loc[i][1] - cy
        loc = np.array([rx, ry])        
        
        #set rf_loc[i,0] (x) rf_loc[i,1] (y) appropriately.
        rf_loc[i,0] = cx
        rf_loc[i,1] = cy
        
        #place proper gaussian in rf_coeff[i]
        rf_coeff[0,i] = gausskernel(k_width,loc, k_width*sigma_ratio)
        rf_coeff[0,i] /= np.sum(rf_coeff[0,i])
        #but why does sumitha divied the gaussian by its sum? Ah to have a 'valid'
        #clipped gaussian that sums to 1?
          
    return rf_loc, rf_coeff

"""
New rf gen function. Striving to maintain backwards compatibility
@tessellation is the raw node locations [x,y] array
@kernel_ratio is the ratio of kernel to sigma (2-3 is fine?)
@sigma_base is the base sigma, or global sigma scaling factor
@sigma_power is the power term applied to sigma scaling with eccentricity
@min_rf is the same as in sumitha's code

NOTES: 
- min_rf is actually min_rf here, not mean_rf as in sumitha's fn. Mean might be 
more robust tho, so will see how it turns out.
- min_rf imposed over 20 foveal nodes, not 5
- sigma_power added, together with a necessary correction
- scaling dist_5 with the min_rf rule has a strong effect on output
"""
def rf_ozimek(tessellation, kernel_ratio, sigma_base, sigma_power, min_rf,dog=False):
    print "rf generation - might take a while..."
    #rf_loc structure; [x, y, 0, 0, 0, rf_sigma, rf_width]
    """ How about making rf_loc[2] equal to eccentricirty, [3] = dist_5 
    and making k_width = max(3, 1.2*dist_5)? This makes k_width always sane.""" 
    rf_loc = np.zeros((len(tessellation), 7))
    rf_coeff = np.ndarray((1, len(tessellation)),dtype='object')
    
    neighbourhood = 6 #5 closest nodes
    
    #compute node density metric dist_5
    d = distance.cdist(tessellation, tessellation)
    s = np.sort(d)
    dist_5 = np.mean(s[:,1:neighbourhood], 1)
    
    #compute min dist_5 for most central 20 nodes
    fov_dist_5 = np.min(dist_5[:20])
    
    #set fov_dist_5 to min_rf (impose min_rf parameter)
    rf_loc[:,:2] = tessellation*(1/fov_dist_5)*min_rf
    
    #Adjust dist_5 to reflect new scale
    dist_5 = dist_5*(1/fov_dist_5)*min_rf

    print "1/3"    
    
    ##determine sigmas
    #sigma_power decreases foveal nodes if dist_5 < 1.0 there, so correct it
#    correction = 1-min_rf if min_rf < 1 else 0
    correction = 0
    rf_loc[:,5] = sigma_base * (dist_5+correction)**sigma_power
    
    print "2/3"
    
    #determine kernel widths
    rf_loc[:,6] = np.ceil(1/kernel_ratio*rf_loc[:,5])
    
    #Use the same method as Sumitha for having even/odd kernels [compatibility]
    for i in range(len(tessellation)):
        k_width = int(rf_loc[i,6])     
        cx, cy = xy_sumitha(rf_loc[i,0], rf_loc[i,1], k_width)
        
        #Obtain subpixel accurate offsets from kernel centre
        rx = rf_loc[i][0] - cx
        ry = rf_loc[i][1] - cy
        loc = np.array([rx, ry]) 
        
        #Set x and y
        rf_loc[i,0] = cx
        rf_loc[i,1] = cy
        
        if dog:
            dog_sigma = 1.6
            rf_coeff[0,i] = gausskernel(k_width, loc, dog_sigma*rf_loc[i,5])
            
        else:
            rf_coeff[0,i] = gausskernel(k_width, loc, rf_loc[i,5])

        rf_coeff[0,i] /= np.sum(rf_coeff[0,i]) # area under curve ->1
        
    print "3/3"
    return rf_loc, rf_coeff, dist_5