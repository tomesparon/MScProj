"""
@author: Lorinc Balog

https://github.com/lorincbalog/RetinaCUDA
"""
import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import ctypes

if sys.platform.startswith('linux'):
    lib = ctypes.cdll.LoadLibrary('../bin/Linux/libRetinaCUDA.so')
elif sys.platform.startswith('win'):
    lib = ctypes.cdll.LoadLibrary('RetinaCUDA.dll')

def resolveError(err):
    if err == -1:
        raise Exception("Invalid arguments")
    elif err == 1:
        raise Exception("Cortex was not initialized properly")
    elif err == 2:
        raise Exception("Cortex size did not match the parameter")
    elif err == 3:
        raise Exception("Image parameteres did not match")

class Cortex(object):
    def __init__(self):
        lib.Cortex_new.argtypes = []
        lib.Cortex_new.restype = ctypes.c_void_p
        lib.Cortex_delete.argtypes = [ctypes.c_void_p]
        lib.Cortex_delete.restype = ctypes.c_void_p

        lib.Cortex_initFromRetinaFields.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.Cortex_initFromRetinaFields.restype = ctypes.c_int

        lib.Cortex_getAlpha.argtypes = [ctypes.c_void_p]
        lib.Cortex_getAlpha.restype = ctypes.c_float
        lib.Cortex_setAlpha.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.Cortex_setAlpha.restype = ctypes.c_void_p

        lib.Cortex_getShrink.argtypes = [ctypes.c_void_p]
        lib.Cortex_getShrink.restype = ctypes.c_float
        lib.Cortex_setShrink.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.Cortex_setShrink.restype = ctypes.c_void_p
    
        lib.Cortex_getRGB.argtypes = [ctypes.c_void_p]
        lib.Cortex_getRGB.restype = ctypes.c_bool
        lib.Cortex_setRGB.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Cortex_setRGB.restype = ctypes.c_void_p

        lib.Cortex_getCortImageX.argtypes = [ctypes.c_void_p]
        lib.Cortex_getCortImageX.restype = ctypes.c_uint
        lib.Cortex_getCortImageY.argtypes = [ctypes.c_void_p]
        lib.Cortex_getCortImageY.restype = ctypes.c_uint
        lib.Cortex_setCortImageSize.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        lib.Cortex_setCortImageSize.restype = ctypes.c_void_p

        lib.Cortex_getLeftSize.argtypes = [ctypes.c_void_p]
        lib.Cortex_getLeftSize.restype = ctypes.c_size_t
        lib.Cortex_setLeftCortexFields.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.Cortex_setLeftCortexFields.restype = ctypes.c_int

        lib.Cortex_getRightSize.argtypes = [ctypes.c_void_p]
        lib.Cortex_getRightSize.restype = ctypes.c_size_t
        lib.Cortex_setRightCortexFields.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.Cortex_setRightCortexFields.restype = ctypes.c_int

        lib.Cortex_setLeftCortexLocations.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Cortex_setLeftCortexLocations.restype = ctypes.c_int

        lib.Cortex_setRightCortexLocations.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Cortex_setRightCortexLocations.restype = ctypes.c_int

        lib.Cortex_getGaussKernelWidth.argtypes = [ctypes.c_void_p]
        lib.Cortex_getGaussKernelWidth.restype = ctypes.c_size_t
        lib.Cortex_getGaussSigma.argtypes = [ctypes.c_void_p]
        lib.Cortex_getGaussSigma.restype = ctypes.c_float
        lib.Cortex_setGauss100.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_float, ctypes.POINTER(ctypes.c_double)]
        lib.Cortex_setGauss100.restype = ctypes.c_int

        lib.Cortex_cortImageLeft.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,\
                                             ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, \
                                             ctypes.c_bool, ctypes.POINTER(ctypes.c_double)]
        lib.Cortex_cortImageLeft.restype = ctypes.c_int

        lib.Cortex_cortImageRight.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,\
                                             ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, \
                                             ctypes.c_bool, ctypes.POINTER(ctypes.c_double)]
        lib.Cortex_cortImageRight.restype = ctypes.c_int

        self.obj = lib.Cortex_new()

    def __del__(self):
        lib.Cortex_delete(self.obj)
    
    @property
    def left_size(self):
        return lib.Cortex_getLeftSize(self.obj)

    @property
    def right_size(self):
        return lib.Cortex_getRightSize(self.obj)
        
    @property
    def cort_image_size(self):
        return [lib.Cortex_getCortImageY(self.obj), lib.Cortex_getCortImageX(self.obj)]
    @cort_image_size.setter
    def cort_image_size(self, size):
        lib.Cortex_setCortImageSize(self.obj, size[1], size[0])
    
    @property
    def rgb(self):
        return lib.Cortex_getRGB(self.obj)
    @rgb.setter
    def rgb(self, value):
        lib.Cortex_setRGB(self.obj, value)
    
    @property
    def alpha(self):
        return lib.Cortex_getAlpha(self.obj)
    @alpha.setter
    def alpha(self, value):
        lib.Cortex_setAlpha(self.obj, value)

    @property
    def shrink(self):
        return lib.Cortex_getShrink(self.obj)
    @shrink.setter
    def shrink(self, value):
        lib.Cortex_setShrink(self.obj, value)

    @property
    def gauss_kernel_width(self):
        return lib.Cortex_getGaussKernelWidth(self.obj)

    @property
    def gauss_sigma(self):
        return lib.Cortex_getGaussSigma(self.obj)

    def init_from_sampling_fields(self, fields):
        if fields is None:
            lib.Cortex_initFromRetinaFields(self.obj, None, 0)
            return
        if fields.shape[0] == 0:
            return False

        loc1D = fields.flatten()
        err = lib.Cortex_initFromRetinaFields(self.obj, (ctypes.c_float * len(loc1D))(*loc1D), fields.shape[0])
        resolveError(err)
        return True

    def set_left_sampling_fields(self, fields):
        if fields.shape[0] == 0:
            return False

        loc1D = fields.flatten()
        err = lib.Cortex_setLeftCortexFields(self.obj, (ctypes.c_float * len(loc1D))(*loc1D), fields.shape[0])
        resolveError(err)

    def set_right_sampling_fields(self, fields):
        if fields.shape[0] == 0:
            return False

        loc1D = fields.flatten()
        err =lib.Cortex_setRightCortexFields(self.obj, (ctypes.c_float * len(loc1D))(*loc1D), fields.shape[0])
        resolveError(err)


    def set_left_cortex_locations(self, loc):
        if loc.shape[0] == 0:
            return False

        loc1D = loc.flatten()
        err = lib.Cortex_setLeftCortexLocations(self.obj, (ctypes.c_double * len(loc1D))(*loc1D), loc.shape[0])
        resolveError(err)

    def set_right_cortex_locations(self, loc):
        if loc.shape[0] == 0:
            return False

        loc1D = loc.flatten()
        err = lib.Cortex_setRightCortexLocations(self.obj, (ctypes.c_double * len(loc1D))(*loc1D), loc.shape[0])
        resolveError(err)

    def set_gauss100(self, kernel_width=7, sigma=0.8, gauss100=None):
        if gauss100 is None:
             err = lib.Cortex_setGauss100(self.obj, kernel_width, sigma, None)
        else:
            kernel_width = gauss100.shape[2]
            gauss100 = gauss100.flatten()
            err = lib.Cortex_setGauss100(self.obj, kernel_width, sigma, gauss100.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            resolveError(err)

    def cort_image_left(self, image_vector):
        image = np.empty(self.cort_image_size[0] * self.cort_image_size[1] * (3 if self.rgb else 1), dtype=ctypes.c_uint8)

        err = lib.Cortex_cortImageLeft(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(image_vector), \
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.cort_image_size[1], \
            self.cort_image_size[0], self.rgb, None)
        resolveError(err)

        if self.rgb:
            flat_length = self.cort_image_size[0] * self.cort_image_size[1]
            out = np.dstack(\
                (np.resize(image[0:flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[flat_length:2*flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[2*flat_length:3*flat_length], (self.cort_image_size[0], self.cort_image_size[1]))))
        else:
           out = np.resize(image, (self.cort_image_size[0], self.cort_image_size[1]))
        return out
    
    def cort_image_right(self, image_vector):
        image = np.empty(self.cort_image_size[0] * self.cort_image_size[1] * (3 if self.rgb else 1), dtype=ctypes.c_uint8)

        err = lib.Cortex_cortImageRight(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(image_vector), \
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.cort_image_size[1], \
            self.cort_image_size[0], self.rgb, None)
        resolveError(err)

        if self.rgb:
            flat_length = self.cort_image_size[0] * self.cort_image_size[1]
            out = np.dstack(\
                (np.resize(image[0:flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[flat_length:2*flat_length], (self.cort_image_size[0], self.cort_image_size[1])),\
                np.resize(image[2*flat_length:3*flat_length], (self.cort_image_size[0], self.cort_image_size[1]))))
        else:
            out = np.resize(image, (self.cort_image_size[0], self.cort_image_size[1]))
        return out

def create_cortex_from_fields(loc, alpha=15, shrink=0.5, k_width=7, sigma=0.8, gauss100=None, rgb=False):
    # Instantiate cortex
    cort = Cortex()
    # Set parameters. Alpha and shrink should be set before anything else
    # Changing these parameters will invalidate the L_loc and R_loc
    cort.rgb = rgb
    cort.alpha = alpha
    cort.shrink = shrink
    # gauss100 can be generated by cortex_cuda by leaving gauss100=None
    # and provide kernel width and sigma
    # or can be assigned (as in retina): k_width still mandatory
    cort.set_gauss100(k_width, sigma, gauss100)
    # locations and cort img size can be generated by CUDA,
    # from the samplingfields' locations
    cort.init_from_sampling_fields(loc)
    return cort

def create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, cort_img_size, k_width=7, sigma=0.8, gauss100=None, rgb=False):
    # Instantiate cortex
    cort = Cortex()
    cort.rgb = rgb
    #If the user provides L_loc and R_loc there is no need to set alpha and shrink 
    # gauss100 can be generated by cortex_cuda by leaving gauss100=None
    # and provide kernel width and sigma
    # or can be assigned (as in retina): k_width still mandatory
    cort.set_gauss100(k_width, sigma, gauss100)
    # User can provide everything separately
    # IMPORTANT L and R must have all 7 values in a row (instead of 3) 
    # -> cortex.py line 30,32 change [i,:3] to [i,:]
    cort.set_left_sampling_fields(L)
    cort.set_right_sampling_fields(R)
    cort.set_left_cortex_locations(L_loc)
    cort.set_right_cortex_locations(R_loc)
    cort.cort_image_size = cort_img_size
    return cort
