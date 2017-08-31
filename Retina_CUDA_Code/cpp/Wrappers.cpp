#include "Retina.cuh"
#include "Cortex.cuh"
#include <algorithm>
/**
 * These wrapper functions are C wrappers for the underlying CUDA C++ implementation.
 * Python ctypes can only communicate with C functions, so here we go.
 * Since these functions will be exported to a dll, sufficent prefix macros need to be provided
 */

#if defined(_MSC_VER)
    //  Microsoft
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

std::vector<SamplingPoint> spVecFromArrays(float *h_loc, size_t numOfLocs, double *h_coeff = NULL) {
	std::vector<SamplingPoint> v;
	int coeffOffset = 0;
	for (int i = 0; i != numOfLocs; ++i) {
		float *locStart = h_loc + i * 7;
		size_t kernelSize = *(locStart + 6);
		double *coeffStart = h_coeff + coeffOffset;
		SamplingPoint sp(*locStart, *(locStart + 1), *(locStart + 2),
			*(locStart + 3), *(locStart + 4), *(locStart + 5), kernelSize);
		if (h_coeff != NULL)
			sp.setKernel(std::vector<double>(coeffStart, coeffStart + kernelSize * kernelSize));
		v.push_back(sp);
		coeffOffset += kernelSize * kernelSize;
	}
	return v;
}

std::vector<double2> d2VecFromArray(double *h_loc, size_t numOfLocs) {
	std::vector<double2> v;
	for (int i = 0; i != numOfLocs; ++i) {
		double2 d2;
		d2.x = *(h_loc + 2 * i);
		d2.y = *(h_loc + 2 * i + 1);
		v.push_back(d2);
	}
	return v;
}

extern "C" {
/**
 * Wrappers for the Retina class.
 */
	EXPORT
	Retina* Retina_new() { return new Retina(); }
	EXPORT
	void Retina_delete(Retina *ret) { delete ret; }

	EXPORT
	int Retina_setSamplingFields(Retina *ret, float *h_loc, double *h_coeff, size_t numOfLocs) {
		auto tmp = spVecFromArrays(h_loc, numOfLocs, h_coeff);
		return ret->setSamplingFields(tmp.data(), tmp.size());
	}

	EXPORT
	int Retina_getSamplingFields(Retina *ret, float *h_loc, double *h_coeff, size_t retinaSize) {
		return 0;
		/*SamplingPoint *h_points = new SamplingPoint[retinaSize];
		return ret->getSamplingFields(h_points, retinaSize);
		for (int i = 0; i != retinaSize; ++i)
			h_loc[]

		delete [] h_points;*/
	}

	EXPORT
	int Retina_setGaussNormImage(Retina *ret, double *h_gauss = NULL, size_t gaussH = 0,
			   	   	   	   	   	 size_t gaussW = 0, size_t gaussC = 0) {
		return ret->setGaussNormImage(h_gauss, gaussH, gaussW, gaussC);
	}

	EXPORT
	int Retina_getGaussNormImage(Retina *ret, double *h_gauss, size_t gaussH, size_t gaussW, size_t gaussC) {
		return ret->getGaussNormImage(h_gauss, gaussH, gaussW, gaussC);
	}

	EXPORT
	int Retina_sample(Retina *ret, const uchar *h_imageIn, size_t imageH, size_t imageW, size_t imageC,
					  double *h_imageVector, size_t vectorLength, bool keepImageVectorOnDevice = false) {
		return ret->sample(h_imageIn, imageH, imageW, imageC, h_imageVector, vectorLength, keepImageVectorOnDevice);
	}

	EXPORT
	int Retina_inverse(Retina *ret, const double *h_imageVector,  size_t vectorLength,
					   uchar *h_imageInverse, size_t imageH, size_t imageW, size_t imageC,
					   bool useImageVectorOnDevice = false) {
		return ret->inverse(h_imageVector, vectorLength, h_imageInverse, imageH, imageW, imageC, useImageVectorOnDevice);
	}

	EXPORT
	int Retina_getRetinaSize(Retina *ret) { return ret->getRetinaSize(); }

	EXPORT
	int Retina_getImageHeight(Retina *ret) { return ret->getImageHeight(); }
	EXPORT
	void Retina_setImageHeight(Retina *ret, const int imageH) { ret->setImageHeight(imageH); }

	EXPORT
	int Retina_getImageWidth(Retina *ret) { return ret->getImageWidth(); }
	EXPORT
	void Retina_setImageWidth(Retina *ret, const int imageW) { ret->setImageWidth(imageW); }

	EXPORT
	bool Retina_getRGB(Retina *ret) { return ret->getRGB(); }
	EXPORT
	void Retina_setRGB(Retina *ret, const bool rgb) { ret->setRGB(rgb);	}

	EXPORT
	int Retina_getCenterX(Retina *ret) { return ret->getCenterX(); }
	EXPORT
	void Retina_setCenterX(Retina *ret, const int centerX) { ret->setCenterX(centerX); }

	EXPORT
	int Retina_getCenterY(Retina *ret) { return ret->getCenterY(); }
	EXPORT
	void Retina_setCenterY(Retina *ret, const int centerY) { ret->setCenterY(centerY); }

/**
 * Wrappers for the Cortex class.
 */
	EXPORT
	Cortex* Cortex_new() { return new Cortex(); }
	EXPORT
	void Cortex_delete(Cortex *cort) { delete cort; }


	EXPORT
	int Cortex_initFromRetinaFields(Cortex *cort, float *h_loc = NULL, size_t numOfLocs = 0) {
		if (h_loc == NULL || numOfLocs == 0) {
			return cort->initFromCortexFields(nullptr, 0, nullptr, 0);
		}
		std::vector<SamplingPoint> left;
		std::vector<SamplingPoint> right;
		for (int i = 0; i != numOfLocs; ++i) {
			float *locStart = h_loc + i * 7;
			size_t kernelSize = *(locStart + 6);
			SamplingPoint sp(*locStart, *(locStart + 1), i,
			*(locStart + 3), *(locStart + 4), *(locStart + 5), kernelSize);
			sp._x < 0 ? left.push_back(sp) : right.push_back(sp);
		}
		return cort->initFromCortexFields(left.data(), left.size(), right.data(), right.size());
	}

	EXPORT
	float Cortex_getAlpha(Cortex *cort) { return cort->getAlpha(); }
	EXPORT
	void Cortex_setAlpha(Cortex *cort, float alpha) { cort->setAlpha(alpha); }

	EXPORT
	float Cortex_getShrink(Cortex *cort) { return cort->getShrink(); }
	EXPORT
	void Cortex_setShrink(Cortex *cort, float shrink) { cort->setShrink(shrink); }

	EXPORT
	bool Cortex_getRGB(Cortex *cort) { return cort->getRGB(); }
	EXPORT
	void Cortex_setRGB(Cortex *cort, bool rgb) { cort->setRGB(rgb); }

	EXPORT
	uint Cortex_getCortImageX(Cortex *cort) { return cort->getCortImageSize().x; }
	EXPORT
	uint Cortex_getCortImageY(Cortex *cort) { return cort->getCortImageSize().y; }
	EXPORT
	void Cortex_setCortImageSize(Cortex *cort, uint cortImgX, uint cortImgY) {
		uint2 cortImgSize; cortImgSize.x = cortImgX; cortImgSize.y = cortImgY;
		cort->setCortImageSize(cortImgSize);
	}

	EXPORT
	size_t Cortex_getLeftSize(Cortex *cort) { return cort->getLeftSize(); }
	EXPORT
	int Cortex_setLeftCortexFields(Cortex *cort, float *h_fields, size_t numOfFields) {
		auto spv = spVecFromArrays(h_fields, numOfFields);
		return cort->setLeftCortexFields(spv.data(), spv.size());
	}

	EXPORT
	size_t Cortex_getRightSize(Cortex *cort) { return cort->getRightSize(); }
	EXPORT
	int Cortex_setRightCortexFields(Cortex *cort, float *h_fields, size_t numOfFields) {
		auto spv = spVecFromArrays(h_fields, numOfFields);
		return cort->setRightCortexFields(spv.data(), spv.size());
	}

	EXPORT
	int Cortex_setLeftCortexLocations(Cortex *cort, double *h_loc, size_t numOfLocs) {
		auto d2v = d2VecFromArray(h_loc, numOfLocs);
		return cort->setLeftCortexLocations(d2v.data(), d2v.size());
	}

	EXPORT
	int Cortex_setRightCortexLocations(Cortex *cort, double *h_loc, size_t numOfLocs) {
		auto d2v = d2VecFromArray(h_loc, numOfLocs);
		return cort->setRightCortexLocations(d2v.data(), d2v.size());
	}

	EXPORT
	size_t Cortex_getGaussKernelWidth(Cortex *cort) { return cort->getGaussKernelWidth(); }
	EXPORT
	float Cortex_getGaussSigma(Cortex *cort) { return cort->getGaussSigma(); }

	EXPORT
	int Cortex_setGauss100(Cortex *cort, const uint kernelWidth, const float sigma, double *h_gauss = NULL) {
		return cort->setGauss100(kernelWidth, sigma, h_gauss);
	}

	EXPORT
	int Cortex_cortImageLeft(Cortex *cort, double *h_imageVector,  size_t vecLen, uchar *h_result,
				size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = NULL) {
		return cort->cortImageLeft(h_imageVector, vecLen, h_result, cortImgX, cortImgY, rgb, d_imageVector);
	}

	EXPORT
	int Cortex_cortImageRight(Cortex *cort, double *h_imageVector,  size_t vecLen, uchar *h_result,
				size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = NULL) {
		return cort->cortImageRight(h_imageVector, vecLen, h_result, cortImgX, cortImgY, rgb, d_imageVector);
	}
}
