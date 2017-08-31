#include "Cortex.cuh"
#include <iostream>
#include "sm_60_atomic_functions.h"
#include "CUDAHelper.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__constant__ float ALPHA;
__constant__ float SHRINK;
__constant__ uint2 CORT_IMG_SIZE;
__constant__ size_t GAUSS_KERNEL_WIDTH;
__constant__ float GAUSS_SIGMA;

struct add_double2 {
    __device__ double2 operator()(const double2& a, const double2& b) const {
        double2 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
 };

struct min_vals_double2 {
    __device__ double2 operator()(const double2& a, const double2& b) const {
        double2 r;
        r.x = a.x < b.x ? a.x : b.x ;
        r.y = a.y < b.y ? a.y : b.y;
        return r;
    }
 };

struct max_vals_double2 {
    __device__ double2 operator()(const double2& a, const double2& b) const {
        double2 r;
        r.x = a.x < b.x ? b.x : a.x ;
        r.y = a.y < b.y ? b.y : a.y;
        return r;
    }
 };

__device__ double gauss(float sigma, float x, float y, float mean = 0.0) {
	float norm = sqrtf(x*x + y*y);
	return exp(-powf((norm - mean), 2) / (2 * powf(sigma, 2))) / sqrtf(2 * M_PI * powf(sigma, 2));
}

__global__ void cort_map_left_kernel(SamplingPoint *d_leftFields, double2 *d_leftLoc, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	SamplingPoint *point = &d_leftFields[index];
	d_leftLoc[index].y = sqrtf(powf(point->_x - ALPHA, 2) + powf(point->_y, 2));
	double theta = atan2(point->_y, point->_x - ALPHA);
	d_leftLoc[index].x = theta + (theta < 0 ? M_PI : -M_PI);
}

__global__ void cort_map_right_kernel(SamplingPoint *d_rightFields, double2 *d_rightLoc, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	SamplingPoint *point = &d_rightFields[index];
	d_rightLoc[index].y = sqrtf(powf(point->_x + ALPHA, 2) + powf(point->_y, 2));
	d_rightLoc[index].x = atan2(point->_y, point->_x + ALPHA);
}

__global__ void cort_norm_kernel(double *d_norm_img, double2 *d_loc,
		double *d_gauss, size_t locSize, bool rgb) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (locSize <= globalIndex)
		return;

	int channel = globalIndex / (locSize / (rgb ? 3 : 1));
	int offset = channel * CORT_IMG_SIZE.x * CORT_IMG_SIZE.y;
	int index = globalIndex % (locSize / (rgb ? 3 : 1));

	double x = d_loc[index].x;
	double y = d_loc[index].y;

	int dx = (int)(10 * ((round(x * 10) / 10 - round(x))));
	dx < 0 ? dx = 10 + dx : dx;
	int dy = (int)(10 * ((round(y * 10) / 10 - round(y))));
	dy < 0 ? dy = 10 + dy : dy;

	double *kernel = &d_gauss[(dx * 10 + dy) * GAUSS_KERNEL_WIDTH * GAUSS_KERNEL_WIDTH];

	int X = (int)round(x) - GAUSS_KERNEL_WIDTH / 2;
	int Y = (int)round(y) - GAUSS_KERNEL_WIDTH / 2;

	for (int i = 0; i != GAUSS_KERNEL_WIDTH; ++i) {
		for (int j = 0; j != GAUSS_KERNEL_WIDTH; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < CORT_IMG_SIZE.x && Y + i < CORT_IMG_SIZE.y)
				atomicAdd(&d_norm_img[offset + (Y + i) * CORT_IMG_SIZE.x + X + j], kernel[i * GAUSS_KERNEL_WIDTH + j]);
		}
	}
}

__global__ void cort_image_kernel(double *d_img, double *d_img_vector, SamplingPoint *d_fields,
		double2 *d_loc, double *d_gauss, size_t locSize, size_t vecLen, bool rgb) {
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (locSize <= globalIndex)
		return;

	int channel = globalIndex / (locSize / (rgb ? 3 : 1));
	int offset = channel * CORT_IMG_SIZE.x * CORT_IMG_SIZE.y;
	int index = globalIndex % (locSize / (rgb ? 3 : 1));
	int vecOffset = channel * vecLen;

	double x = d_loc[index].x;
	double y = d_loc[index].y;

	int dx = (int)(10 * ((round(x * 10) / 10 - round(x))));
	dx < 0 ? dx = 10 + dx : dx;
	int dy = (int)(10 * ((round(y * 10) / 10 - round(y))));
	dy < 0 ? dy = 10 + dy : dy;

	double *kernel = &d_gauss[(dx * 10 + dy) * GAUSS_KERNEL_WIDTH * GAUSS_KERNEL_WIDTH];

	int X = (int)round(x) - GAUSS_KERNEL_WIDTH / 2;
	int Y = (int)round(y) - GAUSS_KERNEL_WIDTH / 2;

	double value = d_img_vector[vecOffset + d_fields[index]._i];
	for (int i = 0; i != GAUSS_KERNEL_WIDTH; ++i) {
		for (int j = 0; j != GAUSS_KERNEL_WIDTH; ++j) {
			if (X + j >= 0 && Y + i >= 0 && X + j < CORT_IMG_SIZE.x && Y + i < CORT_IMG_SIZE.y)
				atomicAdd(&d_img[offset + (Y + i) * CORT_IMG_SIZE.x + X + j], value * kernel[i * GAUSS_KERNEL_WIDTH + j]);
		}
	}
}

__global__ void cort_prepare_kernel(double2 *d_loc, double2 min, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	d_loc[index].x += GAUSS_KERNEL_WIDTH - min.x;
	d_loc[index].x *= SHRINK;

	d_loc[index].y += GAUSS_KERNEL_WIDTH - min.y;
	d_loc[index].y *= SHRINK;
}

__global__ void euclidean_distance_kernel(double2 *d_loc, double2 *d_out, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size * size <= index)
		return;

	int x = index % size;
	int y = index / size;

	double2 a = d_loc[x];
	double2 b = d_loc[y];

	d_out[index].x = sqrtf(powf((b.x - a.x), 2));
	d_out[index].y = sqrtf(powf((b.y - a.y), 2));
}

__global__ void scale_theta_flip_y_kernel(double2 *d_loc, double norm, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	d_loc[index].x *= norm;
	d_loc[index].y *= -1;
}

__global__ void gauss_kernel(double *gauss100) {
	int index = (blockIdx.x + threadIdx.x * blockDim.x ) * GAUSS_KERNEL_WIDTH * GAUSS_KERNEL_WIDTH;

	float x = blockIdx.x * 0.1;
	float y = threadIdx.x * 0.1;
	float dx = GAUSS_KERNEL_WIDTH / 2 + x;
	float dy = GAUSS_KERNEL_WIDTH / 2 + y;

	for (int i = 0; i != GAUSS_KERNEL_WIDTH; ++i) {
		for (int j = 0; j != GAUSS_KERNEL_WIDTH; ++j) {
			gauss100[index + i * GAUSS_KERNEL_WIDTH + j] = gauss(GAUSS_SIGMA, dx - i, dy - j);
		}
	}
}

__global__ void normalise(uchar *d_norm, double *d_image, double *normaliser, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (size <= index)
		return;

	d_norm[index] = normaliser[index] == 0.0 ? 0 : (int)(d_image[index] / normaliser[index]);
}

template <class T>
void setPointerToNull(T **d_ptr) {
	if (*d_ptr != nullptr){
		cudaFree(*d_ptr);
		cudaCheckErrors("ERROR");
		*d_ptr = nullptr;
	}
}

Cortex::~Cortex() {
	setPointerToNull(&d_leftFields);
	setPointerToNull(&d_rightFields);
	setPointerToNull(&d_leftLoc);
	setPointerToNull(&d_rightLoc);
	setPointerToNull(&d_gauss);
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
}

int Cortex::cortImage(double *h_imageVector, size_t vecLen, double **d_norm, uchar *h_result,
			size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector,
			SamplingPoint *d_fields, double2 *d_loc, size_t locSize) {
	if (!isReady())
		return ERRORS::uninitialized;
	if ((h_imageVector == nullptr && d_imageVector == nullptr) || h_result == nullptr)
		return ERRORS::invalidArguments;
	if (cortImgX != _cortImgSize.x || cortImgY != _cortImgSize.y || rgb != _rgb ||
			vecLen != _channels * (_leftCortexSize + _rightCortexSize))
		return ERRORS::imageParametersDidNotMatch;
	double *d_img;
	cudaMalloc((void**)&d_img, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
	cudaMemset(d_img, 0.0, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
	double *_d_imageVector;
	if (d_imageVector != nullptr)
		_d_imageVector = d_imageVector;
	else {
		cudaMalloc((void**)&_d_imageVector, _channels * (_leftCortexSize + _rightCortexSize) * sizeof(double));
		cudaMemcpy(_d_imageVector, h_imageVector, _channels * (_leftCortexSize + _rightCortexSize) * sizeof(double), cudaMemcpyHostToDevice);
	}

	cort_image_kernel<<<ceil(_channels * locSize / 512.0), 512>>>(d_img, _d_imageVector,
			d_fields, d_loc, d_gauss, _channels * locSize, _leftCortexSize + _rightCortexSize, _rgb);
	//cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	if (*d_norm == nullptr) {
		cudaMalloc((void**)d_norm, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
		cudaMemset(*d_norm, 0.0, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(double));
		cort_norm_kernel<<<ceil(_channels * locSize / 512.0), 512>>>(*d_norm, d_loc, d_gauss, _channels * locSize, _rgb);
		//cudaDeviceSynchronize();
		cudaCheckErrors("ERROR");
	}

	uchar *d_normalised;
	cudaMalloc((void**)&d_normalised, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(uchar));
	normalise<<<ceil(_channels * _cortImgSize.x * _cortImgSize.y / 512.0), 512>>>(
			d_normalised, d_img, *d_norm, _channels * _cortImgSize.x * _cortImgSize.y);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	cudaMemcpy(h_result, d_normalised, _channels * _cortImgSize.x * _cortImgSize.y * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");

	cudaFree(d_normalised);
	if (d_imageVector == nullptr)
		cudaFree(_d_imageVector);
	cudaFree(d_img);
	return 0;
}

int Cortex::cortImageLeft(double *h_imageVector, size_t vecLen, uchar *h_result,
							size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector) {
	return cortImage(h_imageVector, vecLen, &d_leftNorm, h_result, cortImgX, cortImgY, rgb,
					 d_imageVector, d_leftFields, d_leftLoc, _leftCortexSize);
}

int Cortex::cortImageRight(double *h_imageVector, size_t vecLen, uchar *h_result,
							size_t cortImgX, size_t cortImgY, bool rgb, double *d_imageVector) {
	return cortImage(h_imageVector, vecLen, &d_rightNorm, h_result, cortImgX, cortImgY, rgb,
					 d_imageVector, d_rightFields, d_rightLoc, _rightCortexSize);
}

error Cortex::initFromCortexFields(SamplingPoint *h_leftFields, size_t leftSize,
											SamplingPoint *h_rightFields, size_t rightSize) {

	if (isnan(_shrink) || isnan(_alpha))
		return ERRORS::uninitialized;

	setLeftCortexFields(h_leftFields, leftSize);
	setRightCortexFields(h_rightFields, rightSize);

	if (d_leftFields == nullptr || d_rightFields == nullptr)
		return ERRORS::invalidArguments;

	setPointerToNull(&d_leftLoc);
	cudaMalloc((void**)&d_leftLoc, _leftCortexSize * sizeof(double2));
	cort_map_left_kernel<<<ceil(_leftCortexSize / 512.0), 512>>>(d_leftFields, d_leftLoc, _leftCortexSize);
	//cudaDeviceSynchronize();
	//cudaCheckErrors("ERROR");

	setPointerToNull(&d_rightLoc);
	cudaMalloc((void**)&d_rightLoc, _rightCortexSize * sizeof(double2));
	cort_map_right_kernel<<<ceil(_rightCortexSize / 512.0), 512>>>(d_rightFields, d_rightLoc, _rightCortexSize);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	double2 *d_eucl_left;
	cudaMalloc((void**)&d_eucl_left, _leftCortexSize * _leftCortexSize * sizeof(double2));
	euclidean_distance_kernel<<<ceil(_leftCortexSize * _leftCortexSize / 1024.0), 1024>>>(d_leftLoc, d_eucl_left, _leftCortexSize);
	//cudaDeviceSynchronize();
	//cudaCheckErrors("ERROR");

	double2 *d_eucl_right;
	cudaMalloc((void**)&d_eucl_right, _rightCortexSize * _rightCortexSize * sizeof(double2));
	euclidean_distance_kernel<<<ceil(_rightCortexSize * _rightCortexSize / 1024.0), 1024>>>(d_rightLoc, d_eucl_right, _rightCortexSize);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	thrust::device_ptr<double2> d_leftLoc_begin(d_eucl_left);
	thrust::device_ptr<double2> d_leftLoc_end(d_eucl_left + _leftCortexSize * _leftCortexSize);
	thrust::device_ptr<double2> d_rightLoc_begin(d_eucl_right);
	thrust::device_ptr<double2> d_rightLoc_end(d_eucl_right + _rightCortexSize * _rightCortexSize);

	double2 init; init.x = init.y = 0.0;
	double2 sum_left = thrust::reduce(d_leftLoc_begin, d_leftLoc_end, init, add_double2());

	init.x = init.y = 0.0;
	double2 sum_right = thrust::reduce(d_rightLoc_begin, d_rightLoc_end, init, add_double2());

	double xd = (sum_left.x / (_leftCortexSize * _leftCortexSize) + sum_right.x / (_rightCortexSize * _rightCortexSize)) / 2;
	double yd = (sum_left.y / (_leftCortexSize * _leftCortexSize) + sum_right.y / (_rightCortexSize * _rightCortexSize)) / 2;

	scale_theta_flip_y_kernel<<<ceil(_leftCortexSize / 512.0), 512>>>(d_leftLoc, yd/xd, _leftCortexSize);
	//cudaDeviceSynchronize();
	//cudaCheckErrors("ERROR");

	scale_theta_flip_y_kernel<<<ceil(_rightCortexSize / 512.0), 512>>>(d_rightLoc, yd/xd, _rightCortexSize);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	thrust::device_ptr<double2> d_l_b(d_leftLoc);
	thrust::device_ptr<double2> d_l_e(d_leftLoc + _leftCortexSize);
	init.x = init.y = 10000.0;
	double2 min_l = thrust::reduce(d_l_b, d_l_e, init, min_vals_double2());

	thrust::device_ptr<double2> d_r_b(d_rightLoc);
	thrust::device_ptr<double2> d_r_e(d_rightLoc + _rightCortexSize);
	init.x = init.y = 10000.0;
	double2 min_r = thrust::reduce(d_r_b, d_r_e, init, min_vals_double2());

	cort_prepare_kernel<<<ceil(_leftCortexSize / 512.0), 512>>>(d_leftLoc, min_l, _leftCortexSize);
	//cudaDeviceSynchronize();
	//cudaCheckErrors("ERROR");

	cort_prepare_kernel<<<ceil(_rightCortexSize / 512.0), 512>>>(d_rightLoc, min_r, _rightCortexSize);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");

	init.x = init.y = -10000.0;
	_cortImgSize.x = thrust::reduce(d_l_b, d_l_e, init, max_vals_double2()).x + _gaussKernelWidth / 2;
	_cortImgSize.y = thrust::reduce(d_l_b, d_l_e, init, max_vals_double2()).y + _gaussKernelWidth / 2;

	cudaMemcpyToSymbol(CORT_IMG_SIZE, &_cortImgSize, sizeof(uint2));
	cudaCheckErrors("ERROR");

	cudaFree(d_eucl_left);
	cudaFree(d_eucl_right);

	return 0;
}

void Cortex::gauss100() {
	setPointerToNull(&d_gauss);
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
	cudaMalloc((void**)&d_gauss, 100 * _gaussKernelWidth * _gaussKernelWidth * sizeof(double));
	gauss_kernel<<<10, 10>>>(d_gauss);
	cudaDeviceSynchronize();
	cudaCheckErrors("ERROR");
}

bool Cortex::isReady() const {
	return _leftCortexSize != 0 && _rightCortexSize != 0 &&
			d_leftLoc != nullptr && d_rightLoc != nullptr &&
			_cortImgSize.x != 0 && _cortImgSize.y != 0 &&
			_gaussKernelWidth != 0 && d_gauss != nullptr;
}

void Cortex::setAlpha(float alpha) {
	if (alpha == _alpha)
		return;
	setPointerToNull(&d_leftLoc);
	setPointerToNull(&d_rightLoc);
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
	_alpha = alpha;
	cudaMemcpyToSymbol(ALPHA, &_alpha, sizeof(float));
	cudaCheckErrors("ERROR");
}

void Cortex::setShrink(float shrink) {
	if (shrink == _shrink)
		return;
	setPointerToNull(&d_leftLoc);
	setPointerToNull(&d_rightLoc);
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
	_shrink = shrink;
	cudaMemcpyToSymbol(SHRINK, &_shrink, sizeof(float));
	cudaCheckErrors("ERROR");
}

void Cortex::setRGB(bool rgb) {
	if (rgb == _rgb)
		return;
	_rgb = rgb;
	_channels = _rgb ? 3 : 1;
}

void Cortex::setCortImageSize(uint2 cortImgSize) {
	if (cortImgSize.x == _cortImgSize.x && cortImgSize.y == _cortImgSize.y)
		return;
	_cortImgSize = cortImgSize;
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);
	cudaMemcpyToSymbol(CORT_IMG_SIZE, &_cortImgSize, sizeof(uint2));
	cudaCheckErrors("ERROR");
}

error Cortex::getLeftCortexFields(SamplingPoint *h_leftFields, size_t leftSize) const {
	return getFromDevice(h_leftFields, leftSize, d_leftFields, _leftCortexSize);
}

error Cortex::setLeftCortexFields(const SamplingPoint *h_leftFields, const size_t leftSize) {
	return setOnDevice(h_leftFields, leftSize, &d_leftFields, _leftCortexSize);
}

error Cortex::getRightCortexFields(SamplingPoint *h_rightFields, size_t rightSize) const {
	return getFromDevice(h_rightFields, rightSize, d_rightFields, _rightCortexSize);
}

error Cortex::setRightCortexFields(const SamplingPoint *h_rightFields, size_t rightSize) {
	return setOnDevice(h_rightFields, rightSize, &d_rightFields, _rightCortexSize);
}

error Cortex::getLeftCortexLocations(double2 *h_leftLoc, size_t leftSize) const {
	return getFromDevice(h_leftLoc, leftSize, d_leftLoc, _leftCortexSize);
}

int Cortex::setLeftCortexLocations(const double2 *h_leftLoc, size_t leftSize) {
	if (leftSize != _leftCortexSize)
		return ERRORS::cortexSizeDidNotMatch;
	int err = setOnDevice(h_leftLoc, leftSize, &d_leftLoc, _leftCortexSize);
	if (err == 0) {
		setPointerToNull(&d_leftNorm);
	}
	return err;
}

error Cortex::getRightCortexLocations(double2 *h_rightLoc, size_t rightSize) const {
	return getFromDevice(h_rightLoc, rightSize, d_rightLoc, _rightCortexSize);
}

int Cortex::setRightCortexLocations(const double2 *h_rightLoc, size_t rightSize) {
	if (rightSize != _rightCortexSize)
		return ERRORS::cortexSizeDidNotMatch;
	int err = setOnDevice(h_rightLoc, rightSize, &d_rightLoc, _rightCortexSize);
	if (err == 0)
		setPointerToNull(&d_rightNorm);
	return err;
}

error Cortex::getGauss100( double *h_gauss, size_t kernelWidth, float sigma) const {
	if (kernelWidth != _gaussKernelWidth || sigma != _gaussSigma)
		return ERRORS::invalidArguments;
	cudaMemcpy(h_gauss, d_gauss, 100 * _gaussKernelWidth * _gaussKernelWidth * sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");
	return 0;
}

error Cortex::setGauss100(const size_t kernelWidth, const float sigma, double *h_gauss) {
	if (kernelWidth == 0)
		return ERRORS::invalidArguments;
	_gaussKernelWidth = kernelWidth;
	cudaMemcpyToSymbol(GAUSS_KERNEL_WIDTH, &_gaussKernelWidth, sizeof(size_t));
	_gaussSigma = sigma;
	cudaMemcpyToSymbol(GAUSS_SIGMA, &_gaussSigma, sizeof(float));
	cudaCheckErrors("ERROR");
	setPointerToNull(&d_leftNorm);
	setPointerToNull(&d_rightNorm);

	if (h_gauss == nullptr) {
		gauss100();
	} else {
		setPointerToNull(&d_gauss);
		cudaMalloc((void**)&d_gauss, 100 * _gaussKernelWidth * _gaussKernelWidth * sizeof(double));
		cudaMemcpy(d_gauss, h_gauss, 100 * _gaussKernelWidth * _gaussKernelWidth * sizeof(double), cudaMemcpyHostToDevice);
		cudaCheckErrors("ERROR");
	}
	return 0;
}

template <class T>
error Cortex::getFromDevice(T *h_ptr, const size_t h_size, const T *d_ptr, const size_t d_size) const {
	if (h_ptr == nullptr || h_size == 0)
		return ERRORS::invalidArguments;
	if (h_size != d_size)
		return ERRORS::cortexSizeDidNotMatch;
	if (d_ptr == nullptr)
		return ERRORS::uninitialized;
	cudaMemcpy(h_ptr, d_ptr, sizeof(T) * d_size, cudaMemcpyDeviceToHost);
	cudaCheckErrors("ERROR");
	return 0;
}

template <class T>
error Cortex::setOnDevice(const T *h_ptr, size_t h_size, T **d_ptr, size_t &d_size) {
	if (h_ptr == nullptr || h_size == 0)
		return ERRORS::invalidArguments;

	setPointerToNull(d_ptr);
	cudaMalloc((void**)d_ptr, sizeof(T) * h_size);
	cudaMemcpy(*d_ptr, h_ptr, sizeof(T) * h_size, cudaMemcpyHostToDevice);
	d_size = h_size;
	cudaCheckErrors("ERROR");
	return 0;
}
