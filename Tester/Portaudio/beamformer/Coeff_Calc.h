#ifndef COEFF_CALC_H
#define COEFF_CALC_H

#include "definitions.h"
#include <complex>
//#include <intrin.h>
#include <fftw3.h>
#include <cufft.h>

//Init cubic spline interpolation, precalculation of coefficients. Assumption: fs=1
__global__
void spline_init(cufftComplex* y, float dx, std::size_t signal_length, float* coeff1, float* coeff2, float* coeff3, float* coeff4, float* cp, float* dp, float* d, float* sigma);

//void spline_init(fftwf_complex* y, float dx, std::size_t signal_length, float* coeff1, float* coeff2, float* coeff3, float* coeff4);

#endif