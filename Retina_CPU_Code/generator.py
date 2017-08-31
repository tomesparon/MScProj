# -*- coding: utf-8 -*-
"""

Generates and saves the retinal field from Piotr's rf_generation code.

@author: Tom Esparon
"""
import cv2
import retina
import matplotlib.pyplot as plt
import scipy.ndimage
import os
import scipy.io

import rf_generation4

######### Input parameters
# retina_path: Change tesselation to be used in generation
# 
# kratio is the ratio of kernel to sigma
# sbase is the base sigma, or global sigma scaling factor
# spower is the power term applied to sigma scaling with eccentricity
# min_rf is the minimum size of the retinal field.
min_rf = 1.5 
kratio = 0.2
sbase = 0.4
spower = 1.0
retina_path = os.getcwd() + os.sep + 'ref\\tess256.mat'
##########


# Get key from mat file
matkey = scipy.io.whosmat(retina_path)
key = [(tup[0]) for tup in matkey]
keystring = str(key)[2:-2]
print keystring

# load Mat file
tess = scipy.io.loadmat(retina_path)[keystring]


def preview():
    stdimg_dir = os.getcwd() + os.sep + 'testimage\\'
    print "Using " + os.listdir(stdimg_dir)[0]

    name = os.listdir(stdimg_dir)[0]

    standard_image = cv2.imread(stdimg_dir+name, )
    img = cv2.normalize(standard_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = cv2.cvtColor(standard_image, cv2.COLOR_BGR2GRAY)
    x, y = img.shape[1]/2, img.shape[0]/2
    size = img.shape



    oz_V = retina.sample(img, x, y, ozimek_coeff, ozimek_loc, rgb=False)
    oz_GI = retina.gauss_norm_img(x, y, ozimek_coeff, ozimek_loc, imsize = size)
    oz_I = retina.inverse(oz_V, x, y, ozimek_coeff, ozimek_loc, oz_GI, imsize = size, rgb=False)
    oz_I_crop = retina.crop(oz_I, x, y, ozimek_loc)
    oz_GI_crop = retina.crop(oz_GI, x, y, ozimek_loc)

    # test application of retinal field
    plt.figure(figsize=(6,6),num="Test application of retinal field")
    plt.axis('off')
    plt.imshow(oz_I_crop, cmap='gray')
    plt.show()

    #heatmap of retinal field
    plt.figure(figsize=(6,6),num="Heatmap of retina")
    plt.axis('off')
    plt.imshow(oz_GI_crop, cmap='RdBu')
    plt.show()


# Generate normal coefficients and locations
ozimek_loc, ozimek_coeff, oz_d5 = rf_generation4.rf_ozimek(tess, kratio, sbase, spower, min_rf,dog=False)
scipy.io.savemat(keystring + 'coeff.mat', mdict={keystring: ozimek_coeff})
scipy.io.savemat(keystring + 'loc.mat', mdict={keystring: ozimek_loc})
preview()
# Generate DoG coeeficients and locations
ozimek_loc, ozimek_coeff, oz_d5 = rf_generation4.rf_ozimek(tess, kratio, sbase, spower, min_rf,dog=True)
scipy.io.savemat(keystring + 'coeffdog.mat', mdict={keystring: ozimek_coeff})
scipy.io.savemat(keystring + 'locdog.mat', mdict={keystring: ozimek_loc})
preview()