#ifndef BEAMFORMING_H
#define BEAMFORMER_H

#include "definitions.h"
#include <cufft.h>

//__global__
//void interpolateChannels(const cufftDoubleComplex* inputBuffer, cufftDoubleComplex* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta);

__global__
void interpolateChannels(const double* time_stamps, double dt, cufftDoubleComplex* summedSignals, const int i,\
						 const double* coeff1, const double* coeff2, const double* coeff3, const double* coeff4, double* mus, double* mus2, double* mus3);

//__global__ 
//void beamforming(const cufftDoubleComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftDoubleComplex* summedSignals);

__global__ 
void beamforming(const double* delays, const double* coeff1, const double* coeff2, const double* coeff3, const double* coeff4, cufftDoubleComplex* summedSignals, double* mus, double* mus2, double* mus3);

#endif