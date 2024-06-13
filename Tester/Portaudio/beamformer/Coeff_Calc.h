#ifndef COEFF_CALC_H
#define COEFF_CALC_H

#include "definitions.h"
#include <complex>
//#include <intrin.h>
#include <fftw3.h>
#include <cufft.h>

//Init cubic spline interpolation, precalculation of coefficients. Assumption: fs=1
__global__
void spline_init(cufftDoubleComplex* y, double dx, std::size_t signal_length, double* coeff1, double* coeff2, double* coeff3, double* coeff4, double* cp, double* dp, double* d, double* sigma);

//void spline_init(fftwf_complex* y, double dx, std::size_t signal_length, double* coeff1, double* coeff2, double* coeff3, double* coeff4);

#endif