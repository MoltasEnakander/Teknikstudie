#pragma once
#include <complex>
//#include <intrin.h>

//Cubic spline interpolation, using coefficients precalculated in interp_init. assuming fs=1
void splinter(float dt, std::size_t signal_length, std::complex<float>* coeff1, std::complex<float>* coeff2, std::complex<float>* coeff3, std::complex<float>* coeff4, float* values, std::size_t values_length, std::complex<float>* output);
