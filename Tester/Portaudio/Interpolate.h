#pragma once
#include <complex>
#include <intrin.h>

//Cubic spline interpolation, using coefficients precalculated in interp_init. assuming fs=1
void splinter(double dt, std::size_t signal_length, std::complex<double>* coeff1, std::complex<double>* coeff2, std::complex<double>* coeff3, std::complex<double>* coeff4, double* values, std::size_t values_length, std::complex<double>* output);
