"""

Model for colour opponency and DoG retinal cells
@author: Tom Esparon
"""
import numpy as np
import cv2

def opponency(C, S, theta):
    """Summary
    Function to generate all modified imagevectors that exhibit single colour opponency
    Args:
        C (vector): The sharp retina is passed to the function, 
                    representing the centre of the receptive field
        S (vector): The blurred retina is passed to the function, 
                    representing the surround of the receptive field
        theta (float): Used to filter imagevector values above and below a threshold
    
    Returns:
        scentre: Return the positive rectified imagevector
        ssurr: Return the negative rectified imagevector
    """
    # Slices of the B G R channels of the input imagevectors
    bluecentre = C[:,0]
    greencentre = C[:,1]
    redcentre = C[:,2]
    bluesurr = S[:,0]
    greensurr = S[:,1]
    redsurr = S[:,2]
    # empty float array to store the generated imagevectors
    diff = np.empty((bluecentre.shape[0],8), dtype='float')
    # Generate each species
    diff[:,0] =  (redcentre - greensurr)/(redcentre + greensurr) #R-ON CENTRE/ G-OFF SURROUND
    diff[:,1] = (greencentre - redsurr)/(greencentre + redsurr) #  G-ON CENTRE/ R-OFF SURROUND
    diff[:,2] = (greensurr - redcentre)/(greensurr + redcentre) #  R-OFF CENTRE/ G-ON SURROUND
    diff[:,3] = (redsurr - greencentre)/(redsurr + greencentre) #  G-OFF CENTRE/ R-ON SURROUND
    diff[:,4] =  (bluecentre - (greensurr+redsurr)/2)/(bluecentre + (greensurr+redsurr)/2)#B-ON CENTRE/ Y-OFF SURROUND
    diff[:,5] =  ((greencentre+redcentre)/2 - bluesurr)/((greencentre+redcentre)/2 + bluesurr)#Y-ON CENTRE/ B-OFF SURROUND
    diff[:,6] =  ((greensurr+redsurr)/2 - bluecentre)/((greensurr+redsurr)/2 + bluecentre)#B-OFF CENTRE/ Y-On SURROUND
    diff[:,7] = (bluesurr - (greencentre+redcentre)/2)/(bluesurr + (greencentre+redcentre)/2)#Y-OFF CENTRE/ B-On SURROUND

    # make two copies of the array for further modification
    centre = diff.copy()
    surr = diff.copy()
    # Zero areas that fall outside of threshold
    centre[np.where(centre <= -theta)] = 0.0
    surr[np.where(surr >= theta)] = 0.0
    # add back channels
    tricentre,trisurr = addchannels(centre,surr)
    # make sure scaling is correct
    scentre,ssurr = scaling(tricentre,trisurr)
    return scentre,ssurr


def doubleopponency(C, S, theta):
    """Summary
    Function to generate all modified imagevectors that exhibit double colour opponency
    Args:
        C (vector): The sharp retina is passed to the function, 
                    representing the centre of the receptive field
        S (vector): The blurred retina is passed to the function, 
                    representing the surround of the receptive field
        theta (float): Used to filter imagevector values above and below a threshold
    
    Returns:
        scentre: Return the positive rectified imagevector
        ssurr: Return the negative rectified imagevector
    """
    # Slices of the B G R channels of the input imagevectors
    bluecentre = C[:,0]
    greencentre = C[:,1]
    redcentre = C[:,2]
    bluesurr = S[:,0]
    greensurr = S[:,1]
    redsurr = S[:,2]
    # empty float array to store the generated imagevectors
    diff = np.empty((bluecentre.shape[0],8), dtype='float')
    # Generate each species
    diff[:,0] =  (redcentre - greencentre)/(redcentre + greencentre) 
    diff[:,0]+= (greensurr - redsurr)/(greensurr + redsurr)                          #RG ON CENTRE - GR SURROUND 
    
    diff[:,1] = (greencentre - redcentre)/(greencentre + redcentre) 
    diff[:,1]+= (redsurr - greensurr)/(redsurr + greensurr)                        #  GR ON CENTRE - RG SURROUND
      
    diff[:,2] = (greensurr - redsurr)/(greensurr + redsurr) 
    diff[:,2]+= (redcentre - greencentre)/(redcentre + greencentre)                  #  RG SURROUND -  GR CENTRE
    
    diff[:,3] = (redsurr - greensurr)/(redsurr + greensurr) 
    diff[:,3]+= (greencentre - redcentre)/(greencentre + redcentre)                  #  GR SURROUND - RG CENTRE
    
    diff[:,4] =  (bluecentre - (greencentre+redcentre/2))/(bluecentre + (greencentre+redcentre/2)) 
    diff[:,4]+= ((greensurr+redsurr/2) - bluesurr)/((greensurr+redsurr/2) + bluesurr)    #BY CENTRE - YB SURROUND
    
    diff[:,5] =  ((greencentre+redcentre/2) - bluecentre)/((greencentre+redcentre/2) + bluecentre) 
    diff[:,5]+= (bluesurr - (greensurr+redsurr/2))/(bluesurr + (greensurr+redsurr/2))    #YB CENTRE - BY SURROUND
    
    diff[:,6] =  ((greensurr+redsurr/2) - bluesurr)/(bluesurr + (greensurr+redsurr/2)) 
    diff[:,6]+= (bluecentre - (greencentre+redcentre/2))/(bluecentre + (greencentre+redcentre/2))#BY SURROUND - YB CENTRE
    
    diff[:,7] = (bluesurr - (greensurr+redsurr/2))/(bluesurr + (greensurr+redcentre/2)) 
    diff[:,7]+= ((greencentre+redcentre/2) - bluecentre)/((greencentre+redcentre/2) + bluecentre) #YB SURROUND - BY CENTRE
    # make two copies of the array for further modification
    centre = diff.copy()
    surr = diff.copy()
    # Zero areas that fall outside of threshold
    centre[np.where(centre <= -theta)] = 0.0
    surr[np.where(surr >= theta)] = 0.0
    # add back channels
    tricentre,trisurr = addchannels(centre,surr)
    # make sure scaling is correct
    scentre,ssurr = scaling(tricentre,trisurr)
    return scentre,ssurr


def addchannels(centre,surr):
    """Summary
        Add two more channels to the 1 dimensional imagevector to create a 3 channel imagevector
    Args:
        centre (vector): Imagevector of the positive rectified centre response
        surr (vector): Imagevector of the negative rectified surround response
    
    Returns:
        pdiff: 3D imagevector of the positive rectified centre response
        ndiff: 3D imagevector of the negative rectified surround response
    """
    #Empty 3d arrays of the same dimensions as the input arrays
    pdiff = np.empty((centre.shape[0],centre.shape[1],3), dtype='float')
    ndiff = np.empty((surr.shape[0],surr.shape[1],3), dtype='float')
    # Array of zeros for the other channels
    zero = np.zeros(centre[:,0].shape)
    
    # Place the correct colour into the right channels
    pzero = np.stack((zero,zero,centre[:,0]),axis=-1)
    nzero = np.stack((zero,surr[:,0],zero),axis=-1)
    ptwo = np.stack((zero,zero,centre[:,2]),axis=-1)
    ntwo = np.stack((zero,surr[:,2],zero),axis=-1)

    pone = np.stack((zero,centre[:,1],zero),axis=-1)
    none = np.stack((zero,zero,surr[:,1]),axis=-1)
    pthree = np.stack((zero,centre[:,3],zero),axis=-1)
    nthree = np.stack((zero,zero,surr[:,3]),axis=-1)

    pfour = np.stack((centre[:,4],zero,zero),axis=-1)
    nfour = np.stack((zero,surr[:,4],surr[:,4]),axis=-1)
    psix = np.stack((centre[:,6],zero,zero),axis=-1)
    nsix = np.stack((zero,surr[:,6],surr[:,6]),axis=-1)

    pfive = np.stack((zero,centre[:,5],centre[:,5]),axis=-1)
    nfive = np.stack((surr[:,5],zero,zero),axis=-1)
    psev = np.stack((zero,centre[:,7],centre[:,7]),axis=-1)
    nsev = np.stack((surr[:,7],zero,zero),axis=-1)

    # place right species in order
    pdiff[:,0,:] = pzero
    ndiff[:,0,:] = nzero
    pdiff[:,1,:] = pone
    ndiff[:,1,:] = none
    pdiff[:,2,:] = ptwo
    ndiff[:,2,:] = ntwo
    pdiff[:,3,:] = pthree
    ndiff[:,3,:] = nthree
    pdiff[:,4,:] = pfour
    ndiff[:,4,:] = nfour
    pdiff[:,5,:] = pfive
    ndiff[:,5,:] = nfive
    pdiff[:,6,:] = psix
    ndiff[:,6,:] = nsix
    pdiff[:,7,:] = psev
    ndiff[:,7,:] = nsev

    return pdiff,ndiff

def nonopponency(C,S,theta):
    """Summary
   Function to generate modified imagevectors that exhibit spatiotemporal ganglion cell response
    Args:
        C (vector): The sharp retina is passed to the function, 
                    representing the centre of the receptive field
        S (vector): The blurred retina is passed to the function, 
                    representing the surround of the receptive field
        theta (float): Used to filter imagevector values above and below a threshold
    
    Returns:
        pdiff: Return the positive rectified imagevector
        ndiff: Return the negative rectified imagevector
    """
    redcentre = C[:,2]
    redsurr = S[:,2]
    greencentre = C[:,1]
    greensurr = S[:,1]

    pdiff = np.empty((redcentre.shape[0],3), dtype='float')
    ndiff = np.empty((redcentre.shape[0],3), dtype='float')
    # zero = np.zeros(redcentre.shape)

    rgcentre = (redcentre + greencentre)
    rgsurr = (redsurr + greensurr)
    diff =  (rgcentre - rgsurr)/(rgcentre + rgsurr)
    centre = diff.copy() 
    surr = diff.copy()
    centre[np.where(centre <= -theta)] = 0.0
    surr[np.where(surr >= theta)] = 0.0

    pdiff = centre
    ndiff = surr
    # correct scaling
    pdiff,ndiff = scaling(pdiff,ndiff)
    return pdiff,ndiff

def scaling(centre,surr):
    """Summary
    
    Args:
        centre (vector): Pass an imagevector for centre response
        surr (vector): Pass an imagevector for surround response
    Returns:
        scentre (vector): Return scaled imagevector for centre response
        ssurr (vector): Return scaled imagevector for surround response
    """
    ncentre = cv2.normalize(centre, None, 0.0, 1.0, cv2.NORM_MINMAX)
    nsurr = cv2.normalize(surr, None, -1.0,0.0, cv2.NORM_MINMAX)
    scentre = cv2.convertScaleAbs(ncentre, alpha=255)
    ssurr = cv2.convertScaleAbs(nsurr, alpha=255)
    return scentre,ssurr

