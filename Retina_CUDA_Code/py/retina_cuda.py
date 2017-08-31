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
        raise Exception("Retina was not initialized properly")
    elif err == 2:
        raise Exception("Retina size did not match the parameter")
    elif err == 3:
        raise Exception("Image parameteres did not match")


class Retina(object):
    def __init__(self):
        lib.Retina_new.argtypes = []
        lib.Retina_new.restype = ctypes.c_void_p
        lib.Retina_delete.argtypes = [ctypes.c_void_p]
        lib.Retina_delete.restype = ctypes.c_void_p

        lib.Retina_setSamplingFields.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Retina_setSamplingFields.restype = ctypes.c_int

        '''
        lib.Retina_getSamplingFields.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        lib.Retina_getSamplingFields.restype = ctypes.c_int
        '''
        
        lib.Retina_setGaussNormImage.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.Retina_setGaussNormImage.restype = ctypes.c_int
        
        lib.Retina_getGaussNormImage.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.Retina_getGaussNormImage.restype = ctypes.c_int
        
        lib.Retina_sample.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_bool]
        lib.Retina_sample.restype = ctypes.c_int

        lib.Retina_inverse.argtypes = [ctypes.c_void_p, \
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8), \
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool]
        lib.Retina_inverse.restype = ctypes.c_int
        
        lib.Retina_getRetinaSize.argtypes = [ctypes.c_void_p]
        lib.Retina_getRetinaSize.restype = ctypes.c_int

        lib.Retina_getImageHeight.argtypes = [ctypes.c_void_p]
        lib.Retina_getImageHeight.restype = ctypes.c_int
        lib.Retina_setImageHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setImageHeight.restype = ctypes.c_void_p

        lib.Retina_getImageWidth.argtypes = [ctypes.c_void_p]
        lib.Retina_getImageWidth.restype = ctypes.c_int
        lib.Retina_setImageWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setImageWidth.restype = ctypes.c_void_p

        lib.Retina_getRGB.argtypes = [ctypes.c_void_p]
        lib.Retina_getRGB.restype = ctypes.c_bool
        lib.Retina_setRGB.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.Retina_setRGB.restype = ctypes.c_void_p

        lib.Retina_getCenterX.argtypes = [ctypes.c_void_p]
        lib.Retina_getCenterX.restype = ctypes.c_int
        lib.Retina_setCenterX.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setCenterX.restype = ctypes.c_void_p
        
        lib.Retina_getCenterY.argtypes = [ctypes.c_void_p]
        lib.Retina_getCenterY.restype = ctypes.c_int
        lib.Retina_setCenterY.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.Retina_setCenterY.restype = ctypes.c_void_p


        self.obj = lib.Retina_new()

    def __del__(self):
        lib.Retina_delete(self.obj)

    @property
    def retina_size(self):
        '''int, number of sampling fields in the retina'''
        return lib.Retina_getRetinaSize(self.obj)

    @property
    def image_height(self):
        '''int, height of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getImageHeight(self.obj)
    @image_height.setter
    def image_height(self, value):
        '''int, height of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setImageHeight(self.obj, value)

    @property
    def image_width(self):
        '''int, width of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getImageWidth(self.obj)
    @image_width.setter
    def image_width(self, value):
        '''int, width of the image the retina can process (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setImageWidth(self.obj, value)

    @property
    def rgb(self):
        '''bool, whether the retina can process rgb images (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getRGB(self.obj)
    @rgb.setter
    def rgb(self, value):
        '''bool, whether the retina can process rgb images (input image)
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setRGB(self.obj, value)

    @property
    def center_x(self):
        '''int, X coordinate of the retina center
        Note: in openCV this is [1]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getCenterX(self.obj)
    @center_x.setter
    def center_x(self, value):
        '''int, X coordinate of the retina center
        Note: in openCV this is [1]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setCenterX(self.obj, value)

    @property
    def center_y(self):
        '''int, Y coordinate of the retina center
        Note: in openCV this is [0]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_getCenterY(self.obj)
    @center_y.setter
    def center_y(self, value):
        '''int, Y coordinate of the retina center
        Note: in openCV this is [0]
        Setting the property will invalidate gauss norm image'''
        return lib.Retina_setCenterY(self.obj, value)
    
    def set_samplingfields(self, loc, coeff):
        '''
        Sets the sampling fields of the retina\n
        Parameters
        ----------
        loc : 2D np.array
            7 values each line, locations of the fields (from matlab)
        coeff : kernels of the sampling
        '''
        if loc.shape[0] != len(coeff.flatten()):
            print "Number of locs and coeffs must be the same"
            return
        loc1D = loc.flatten()
        coeff1D = []
        for i in coeff.flatten():
            coeff1D += i.flatten().tolist()

        self.__retina_size = loc.shape[0]
        err = lib.Retina_setSamplingFields(self.obj, (ctypes.c_float * len(loc1D))(*loc1D),
                (ctypes.c_double * len(coeff1D))(*coeff1D), loc.shape[0])
        resolveError(err)

    def set_gauss_norm(self, gauss_norm=None):
        '''
        Sets the gaussian matrix to normalise with on invert\n
        Parameters
        ----------
        guass_norm : 2D np.array, optional
            if None, CUDA will generate the gauss norm
            if not None, height and width must match with retina's 
            (3rd dimension is handled by the function)
        '''
        if gauss_norm is None:
            lib.Retina_setGaussNormImage(self.obj, None, 0, 0, 0)
        else:
            gauss_channels = 1
            gauss_norm_p = gauss_norm.flatten()
            if self.rgb:
                gauss_channels = 3#gauss_norm.shape[2]
                gauss_norm_p = np.vstack((gauss_norm[:,:].flatten(), gauss_norm[:,:].flatten(), gauss_norm[:,:].flatten()))

            err = lib.Retina_setGaussNormImage(self.obj, \
                    gauss_norm_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    gauss_norm.shape[0], gauss_norm.shape[1], gauss_channels)
            resolveError(err)

    def sample(self, image):
        '''
        Sample image\n
        Parameters
        ----------
        image : np.array
            height, width and rgb must match the retina parameters\n
        Returns
        -------
        image_vector : np.array
            sampled flat image vector
            if rgb, must be reshaped to become compatible with Piotr (convert_to_Piotr)
        '''
        image_vector = np.empty(self.retina_size * (3 if self.rgb else 1), dtype=ctypes.c_double)

        image_channels = 1
        image_p = image.flatten()
        if self.rgb:
            image_channels = image.shape[2]
            image_p = np.vstack((image[:,:,0].flatten(), image[:,:,1].flatten(), image[:,:,2].flatten()))
        
        err = lib.Retina_sample(self.obj, image_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), \
                image.shape[0], image.shape[1], image_channels, \
                image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                image_vector.shape[0], False)
        resolveError(err)

        return image_vector

    def inverse(self, image_vector):
        '''
        Invert image from image vector\n
        Parameters
        ----------
        image_vector : np.array
            length must match retina size\n
            if rgb and from Piotr, must be flattened (convert_from_Piotr)
        Returns
        -------
        image : np.array
            inverted image
        '''
        channels = (3 if self.rgb else 1)
        image = np.empty(self.image_height * self.image_width * channels, dtype=ctypes.c_uint8)
        
        err = lib.Retina_inverse(self.obj, \
            image_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
            self.retina_size * channels, image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), \
            self.image_height, self.image_width, channels, False)
        resolveError(err)
        
        if self.rgb:
            flat_length = self.image_height * self.image_width
            out = np.dstack(\
            (np.resize(image[0:flat_length], (self.image_height, self.image_width)),\
            np.resize(image[flat_length:2*flat_length], (self.image_height, self.image_width)),\
            np.resize(image[2*flat_length:3*flat_length], (self.image_height, self.image_width))))
        else:
            out = np.resize(image, (self.image_height, self.image_width))
        return out

def convert_to_Piotr(rgb_image_vector):
    '''
    Reshape flat RGB image vector to become compatible with Piotr's code\n
    Parameters
    ----------
    rgb_image_vector : np.array
        Must be flat, with length of retina_size * 3\n
    Returns
    -------
    rgb_image_vector : np.array
        image vector shaped [retina_size, 3]
    '''
    retina_size = len(rgb_image_vector) / 3
    return np.hstack(\
        (np.resize(rgb_image_vector[0:retina_size], (retina_size,1)),\
        np.resize(rgb_image_vector[retina_size:2*retina_size], (retina_size,1)), \
        np.resize(rgb_image_vector[2*retina_size:3*retina_size], (retina_size,1))))

def convert_from_Piotr(rgb_image_vector):
    '''
    Reshape Piotr's RGB image vector to become compatible CUDA implementation\n
    Parameters
    ----------
    rgb_image_vector : np.array
        must have shape of [retina_size, 3]\n
    Returns
    -------
    rgb_image_vector : np.array
        flattened image vector
    '''
    retina_size = rgb_image_vector.shape[0]
    return np.append(\
    np.resize(rgb_image_vector[:,0], (1, retina_size))[0],\
    [np.resize(rgb_image_vector[:,1], (1, retina_size))[0],\
    np.resize(rgb_image_vector[:,2], (1, retina_size))[0]])

def create_retina(loc, coeff, img_size, center, gauss_norm=None):
    # Instantiate retina
    ret = Retina()
    # Set retina's parameters
    # It is good practice to initialise the retina once
    # and use for multiple sampling and inversion without changin the parameters
    ret.set_samplingfields(loc, coeff) # setting a different samplingfield does not affect the other parameters
    # Changing the parameters below will lead to an invalid gauss norm image (must be reassigned again / generate)
    ret.image_height = img_size[0]

    ret.image_width = img_size[1]
    ret.rgb = (len(img_size) == 3)
    ret.center_x = center[0]
    ret.center_y = center[1]
    # Once image parameters are known,CUDA can generate the gauss norm image (leave the parameter empty or None)
    # or assign a pregenerated one (will check the size)
    ret.set_gauss_norm(gauss_norm)
    # Retina is ready to use
    return ret
