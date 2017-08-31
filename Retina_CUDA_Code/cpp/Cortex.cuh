#ifndef CORTEX__CUH
#define CORTEX__CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "SamplingPoint.cuh"

typedef unsigned short ushort;
typedef unsigned int  uint;
typedef unsigned char uchar;
typedef int error;

class Cortex {
	enum ERRORS {
		invalidArguments = -1,
		uninitialized = 1,
		cortexSizeDidNotMatch,
		imageParametersDidNotMatch
	};

public:
	Cortex() : _rgb(false), _channels(1),
				_leftCortexSize(0), _rightCortexSize(0), _alpha(nanf("")),
				d_leftLoc(nullptr), d_rightLoc(nullptr), d_leftFields(nullptr), d_rightFields(nullptr),
				d_leftNorm(nullptr), d_rightNorm(nullptr), _shrink(nanf("")), _cortImgSize(make_uint2(0,0)),
				_gaussKernelWidth(0), _gaussSigma(nanf("")), d_gauss(nullptr) {}
	Cortex(const Cortex&) = delete;
	~Cortex();

	error initFromCortexFields(SamplingPoint *h_leftFields, size_t leftSize,
										SamplingPoint *h_rightFields, size_t rightSize);

	error cortImageLeft(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr);
	error cortImageRight(double *h_imageVector, size_t vecLen, uchar *h_result,
					size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector = nullptr);

	float getAlpha() const { return _alpha; }
	void setAlpha(float alpha);

	float getShrink() const { return _shrink; }
	void setShrink(float shrink);

	bool getRGB() const { return _rgb; }
	void setRGB(bool rgb);

	uint2 getCortImageSize() const { return _cortImgSize; }
	void setCortImageSize(uint2 cortImgSize);

	size_t getLeftSize() { return _leftCortexSize; }
	error getLeftCortexFields(SamplingPoint *h_leftFields, size_t leftSize) const;
	error setLeftCortexFields(const SamplingPoint *h_leftFields, size_t leftSize);

	size_t getRightSize() { return _rightCortexSize; }
	error getRightCortexFields(SamplingPoint *h_rightFields, size_t rightSize) const;
	error setRightCortexFields(const SamplingPoint *h_rightFields, size_t rightSize);

	error getLeftCortexLocations(double2 *h_leftLoc, size_t leftSize) const;
	error setLeftCortexLocations(const double2 *h_leftLoc, size_t leftSize);

	error getRightCortexLocations(double2 *h_leftLoc, size_t leftSize) const;
	error setRightCortexLocations(const double2 *h_leftLoc, size_t rightSize);

	size_t getGaussKernelWidth() const { return _gaussKernelWidth; }
	float getGaussSigma() { return _gaussSigma; }

	error getGauss100(double *h_gauss, size_t kernelWidth, float sigma) const;
	error setGauss100(const size_t kernelWidth, const float sigma, double *h_gauss = nullptr);

private:
	bool isReady() const;
	void gauss100();
	error cortImage(double *h_imageVector, size_t vecLen, double **d_norm, uchar *h_result,
			size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector,
			SamplingPoint *d_fields, double2 *d_loc, size_t size);
	template <class T>
	error getFromDevice(T *h_fields, const size_t h_size, const T *d_fields, const size_t d_size) const;
	template <class T>
	error setOnDevice(const T *h_fields, const size_t h_size, T **d_fields, size_t &d_size);

	bool _rgb;
	ushort _channels;

	size_t _leftCortexSize;
	size_t _rightCortexSize;
	SamplingPoint *d_leftFields;
	SamplingPoint *d_rightFields;
	double2 *d_leftLoc;
	double2 *d_rightLoc;
	double *d_leftNorm;
	double *d_rightNorm;

	float _alpha;
	float _shrink;
	uint2 _cortImgSize;

	size_t _gaussKernelWidth;
	float _gaussSigma;
	double *d_gauss;
};

#endif //CORTEX__CUH
