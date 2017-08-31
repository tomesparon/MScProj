#include "SamplingPoint.cuh"
#include "Retina.cuh"
#include <iostream>
#include "sm_60_atomic_functions.h"
#include "CUDAHelper.cuh"

__constant__ int IMAGE_H;
__constant__ int IMAGE_W;
__constant__ int CENTER_X;
__constant__ int CENTER_Y;
__constant__ int RETINA_SIZE_R;

__global__ void sample_linear_kernel(uchar *d_in, double *d_image_vector, SamplingPoint *d_points, size_t size) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= globalIndex)
		return;

	int channel = globalIndex / RETINA_SIZE_R;
	int offset = channel * IMAGE_W * IMAGE_H;
	int index = globalIndex % RETINA_SIZE_R;

	SamplingPoint *point = &d_points[index];
	int kernelSize = point->_kernelSize;
	double *kernel = point->d_kernel;

	int X = CENTER_X + point->_x - (float)kernelSize/2 + 0.5;
	int Y = CENTER_Y + point->_y - (float)kernelSize/2 + 0.5;

	double value = 0;
	double normalise =  0;
	for (int i = 0; i != kernelSize; ++i) {
		for (int j = 0; j != kernelSize; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < IMAGE_W && Y + i < IMAGE_H) {
				normalise += kernel[i * kernelSize + j];
				value += (double)d_in[offset + (Y + i) * IMAGE_W + X + j] * kernel[i * kernelSize + j];
			}
		}
	}

	d_image_vector[globalIndex] = normalise != 0 ? value / normalise : 0;
}

__global__ void gaussNorm_kernel(double *d_gauss, SamplingPoint *d_points, size_t size) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= globalIndex)
		return;

	int channel = globalIndex / RETINA_SIZE_R;
	int offset = channel * IMAGE_W * IMAGE_H;
	int index = globalIndex % RETINA_SIZE_R;

	SamplingPoint *point = &d_points[index];
	int kernelSize = point->_kernelSize;
	double *kernel = point->d_kernel;

	int X = CENTER_X + point->_x - (float)kernelSize/2 + 0.5;
	int Y = CENTER_Y + point->_y - (float)kernelSize/2 + 0.5;

	for (int i = 0; i != kernelSize; ++i) {
		for (int j = 0; j != kernelSize; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < IMAGE_W && Y + i < IMAGE_H) {
				atomicAdd(&d_gauss[offset + (Y + i) * IMAGE_W + X + j], kernel[i * kernelSize + j]);
			}
		}
	}
}

__global__ void inverse_kernel(double *d_image_vector, double *d_image_out, SamplingPoint *d_points, size_t size) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= globalIndex)
		return;

	int channel = globalIndex / RETINA_SIZE_R;
	int offset = channel * IMAGE_W * IMAGE_H;
	int index = globalIndex % RETINA_SIZE_R;

	SamplingPoint *point = &d_points[index];
	int kernelSize = point->_kernelSize;
	double *kernel = point->d_kernel;

	int X = CENTER_X + point->_x - (float)kernelSize/2 + 0.5;
	int Y = CENTER_Y + point->_y - (float)kernelSize/2 + 0.5;

	double V = d_image_vector[globalIndex];
	for (int i = 0; i != kernelSize; ++i) {
		for (int j = 0; j != kernelSize; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < IMAGE_W && Y + i < IMAGE_H) {
				atomicAdd(&d_image_out[offset + (Y + i) * IMAGE_W + X + j], V * kernel[i * kernelSize + j]);
			}
		}
	}
}

__global__ void normalise_kernel(double *d_image_out, double *d_gauss, uchar *d_image_out_norm, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	double gauss = d_gauss[index];
	gauss == 0.0 ? d_image_out_norm[index] = 0 : d_image_out_norm[index] = (int)(d_image_out[index] / gauss);
}

template <class T>
void setPointerToNull(T **d_ptr) {
	if (*d_ptr != nullptr){
		cudaFree(*d_ptr);
		cudaCheckErrors("ERROR");
		*d_ptr = nullptr;
	}
}

Retina::~Retina() {
	setPointerToNull(&d_gauss);
	setPointerToNull(&_d_imageVector);
	removeSamplingPointsFromDevice();
}

int Retina::sample(const uchar *h_imageIn, size_t imageH, size_t imageW, size_t imageC,
							double *h_imageVector, size_t vectorLength, bool keepImageVectorOnDevice) {
	if ((h_imageVector == nullptr && !keepImageVectorOnDevice) ||  h_imageIn == nullptr)
		return ERRORS::invalidArguments;
	if (!isReady())
		return ERRORS::uninitialized;
	if (vectorLength != _channels * _retinaSize)
		return ERRORS::retinaSizeDidNotMatch;
	if (!validateImageSize(imageH, imageW, imageC))
		return ERRORS::imageParametersDidNotMatch;
	if (d_points == nullptr || d_gauss == nullptr)
		return ERRORS::uninitialized;

	uchar *d_in;
	cudaMalloc((void**)&d_in, sizeof(uchar) * _channels * _imageH * _imageW);
	cudaMemcpy(d_in, h_imageIn, sizeof(uchar) * _channels * _imageH * _imageW, cudaMemcpyHostToDevice);
	cudaCheckErrors("ERROR");

	double *d_imageVector;
	cudaMalloc((void**)&d_imageVector, _channels * _retinaSize * sizeof(double));
	cudaCheckErrors("ERROR");

	sample_linear_kernel<<<ceil(_channels * _retinaSize / 256.0), 256>>>(d_in, d_imageVector, d_points, _channels * _retinaSize);
	//cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	if (h_imageVector != nullptr) {
		cudaMemcpy(h_imageVector, d_imageVector, sizeof(double) * _channels * _retinaSize, cudaMemcpyDeviceToHost);
		cudaCheckErrors("ERROR");
	}

	cudaFree(d_in);
	cudaCheckErrors("ERROR");
	if (keepImageVectorOnDevice) {
		setPointerToNull(&_d_imageVector);
		_d_imageVector = d_imageVector;
	} else {
		cudaFree(d_imageVector);
		cudaCheckErrors("ERROR");
	}
	return 0;
}

int Retina::inverse(const double *h_imageVector,  size_t vectorLength,
							 uchar *h_imageInverse, size_t imageH, size_t imageW, size_t imageC,
							 bool useImageVectorOnDevice) const {
	if ((h_imageVector == nullptr && !useImageVectorOnDevice) ||  h_imageInverse == nullptr )
		return ERRORS::invalidArguments;
	if (!isReady() || (useImageVectorOnDevice && _d_imageVector == nullptr))
		return ERRORS::uninitialized;
	if (vectorLength != _channels * _retinaSize)
		return ERRORS::retinaSizeDidNotMatch;
	if (!validateImageSize(imageH, imageW, imageC))
		return ERRORS::imageParametersDidNotMatch;

	double *d_imageVector;
	if (useImageVectorOnDevice) {
		d_imageVector = _d_imageVector;
	} else if (h_imageVector != nullptr) {
		cudaMalloc((void**)&d_imageVector, _channels * _retinaSize * sizeof(double));
		cudaMemcpy(d_imageVector, h_imageVector, sizeof(double) * _channels * _retinaSize, cudaMemcpyHostToDevice);
		cudaCheckErrors("ERROR");
	}

	double *d_imageInverse;
	cudaMalloc((void**)&d_imageInverse, sizeof(double) * _channels * _imageH * _imageW);
	cudaMemset(d_imageInverse, 0, sizeof(double) * _channels * _imageH * _imageW);

	inverse_kernel<<<ceil(_channels * _retinaSize / 512.0), 512>>>(d_imageVector, d_imageInverse, d_points, _channels * _retinaSize);
	//cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	uchar *d_imageInverseNorm;
	cudaMalloc((void**)&d_imageInverseNorm, sizeof(uchar) * _channels * _imageH * _imageW);
	normalise_kernel<<<ceil(_channels * _imageW * _imageH / 256.0), 256>>>(d_imageInverse, d_gauss, d_imageInverseNorm, _channels * _imageW * _imageH);
	//cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");


	cudaMemcpy(h_imageInverse, d_imageInverseNorm, sizeof(uchar) * _channels * _imageH * _imageW, cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");

	if (!useImageVectorOnDevice)
		cudaFree(d_imageVector);
	cudaFree(d_imageInverse);
	cudaFree(d_imageInverseNorm);
	cudaCheckErrors("ERROR");
	return 0;
}

int Retina::setSamplingFields(SamplingPoint *h_points, size_t retinaSize) {
	if (h_points == nullptr)
		return ERRORS::invalidArguments;
	removeSamplingPointsFromDevice();
	for (int i = 0; i != retinaSize; ++i) {
		h_points[i].copyToDevice();
	}
	cudaMalloc((void**)&d_points, sizeof(SamplingPoint) * retinaSize);
	cudaMemcpy(d_points, h_points, sizeof(SamplingPoint) * retinaSize, cudaMemcpyHostToDevice);
	cudaCheckErrors("ERROR");

	if (_retinaSize != retinaSize) {
		setPointerToNull(&d_gauss);
		setPointerToNull(&_d_imageVector);
	}

	_retinaSize = retinaSize;
	cudaMemcpyToSymbol(RETINA_SIZE_R, &_retinaSize, sizeof(int));
	cudaCheckErrors("ERROR");
	return 0;
}

int Retina::getSamplingFields(SamplingPoint *h_points, size_t retinaSize) {
	if (retinaSize != _retinaSize)
		return ERRORS::retinaSizeDidNotMatch;
	if (d_points == nullptr && h_points == nullptr)
		return ERRORS::invalidArguments;
	cudaMemcpy(h_points, d_points, sizeof(SamplingPoint) * _retinaSize, cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");
	return 0;
}

int Retina::setGaussNormImage(const double *h_gauss, size_t gaussH, size_t gaussW, size_t gaussC) {
	if (h_gauss != nullptr) {
		if (!validateImageSize(gaussH, gaussW, gaussC))
			return ERRORS::imageParametersDidNotMatch;
		setPointerToNull(&d_gauss);
		cudaMalloc((void**)&d_gauss, sizeof(double) * _channels * _imageH * _imageW);
		cudaMemcpy(d_gauss, h_gauss, sizeof(double) * _channels * _imageH * _imageW, cudaMemcpyHostToDevice);
	} else {
		setPointerToNull(&d_gauss);
		cudaMalloc((void**)&d_gauss, sizeof(double) * _channels * _imageH * _imageW);
		cudaMemset(d_gauss, 0, sizeof(double) * _channels * _imageH * _imageW);
		gaussNorm_kernel<<<ceil(_channels * _retinaSize / 256.0), 256>>>(d_gauss, d_points, _channels * _retinaSize);
		cudaDeviceSynchronize();
		cudaCheckErrors("ERROR");
	}
	return 0;
}

int Retina::getGaussNormImage(double *h_gauss, size_t gaussH, size_t gaussW, size_t gaussC) const {
	if (!validateImageSize(gaussH, gaussW, gaussC))
		return ERRORS::imageParametersDidNotMatch;
	if (d_gauss == nullptr && h_gauss == nullptr)
		return ERRORS::invalidArguments;
	cudaMemcpy(h_gauss, d_gauss, sizeof(double) * _channels * _imageH * _imageW, cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");
	return 0;
}

void Retina::setImageHeight(const int imageH) {
	if (imageH != _imageH)
		setPointerToNull(&d_gauss);
	_imageH = imageH;
	cudaMemcpyToSymbol(IMAGE_H, &imageH, sizeof(int));
	cudaCheckErrors("ERROR");
}

void Retina::setImageWidth(const int imageW) {
	if (imageW != _imageW)
		setPointerToNull(&d_gauss);
	_imageW = imageW;
	cudaMemcpyToSymbol(IMAGE_W, &imageW, sizeof(int));
	cudaCheckErrors("ERROR");
}

void Retina::setRGB(const bool rgb) {
	if (rgb != _rgb)
		setPointerToNull(&d_gauss);
	_rgb = rgb;
	_channels = rgb ? 3 : 1;
}

void Retina::setCenterX(const int centerX) {
	if (centerX != _centerX)
		setPointerToNull(&d_gauss);
	_centerX = centerX;
	cudaMemcpyToSymbol(CENTER_X, &centerX, sizeof(int));
	cudaCheckErrors("ERROR");
}

void Retina::setCenterY(const int centerY) {
	if (centerY != _centerY)
		setPointerToNull(&d_gauss);
	_centerY = centerY;
	cudaMemcpyToSymbol(CENTER_Y, &centerY, sizeof(int));
	cudaCheckErrors("ERROR");
}

double* Retina::imageVectorOnDevice(size_t &vectorLength) {
	vectorLength = _channels * _retinaSize;
	return _d_imageVector;
}

bool Retina::validateImageSize(size_t imageH, size_t imageW, size_t imageC) const {
	if (imageH != _imageH || imageW != _imageW || imageC != _channels)
		return false;
	return true;
}

bool Retina::isReady() const {
	return _imageH != 0 && _imageW != 0 && _centerX != 0 &&
			_centerY != 0 && d_gauss != nullptr && _retinaSize != 0 && d_points != nullptr;
}

int Retina::removeSamplingPointsFromDevice() {
	if (d_points != nullptr) {
		SamplingPoint *h_points = (SamplingPoint*)malloc(sizeof(SamplingPoint) * _retinaSize);
		cudaMemcpy(h_points, d_points, sizeof(SamplingPoint) * _retinaSize, cudaMemcpyDeviceToHost);
		for (int i = 0; i != _retinaSize; ++i)
			h_points[i].removeFromDevice();
		free(h_points);
		setPointerToNull(&d_points);
		cudaCheckErrors("ERROR");
	}
	return 0;
}
