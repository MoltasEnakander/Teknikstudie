#pragma once
#include <complex>
#include <intrin.h>

//Init cubic spline interpolation, precalculation of coefficients. Assumption: fs=1
void spline_init(std::complex<double>* f, double dx, std::size_t signal_length, std::complex<double>* coeff1, std::complex<double>* coeff2, std::complex<double>* coeff3, std::complex<double>* coeff4);
