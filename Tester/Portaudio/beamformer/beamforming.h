#ifndef BEAMFORMING_H
#define BEAMFORMER_H

#include "definitions.h"
#include <cufft.h>

//__global__
//void interpolateChannels(const cufftComplex* inputBuffer, cufftComplex* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta);

__global__
void interpolateChannels(const float* time_stamps, float dt, cufftComplex* summedSignals, const int i,\
						 const float* coeff1, const float* coeff2, const float* coeff3, const float* coeff4, float* mus, float* mus2, float* mus3);

//__global__ 
//void beamforming(const cufftComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftComplex* summedSignals);

__global__ 
void beamforming(const float* delays, const float* coeff1, const float* coeff2, const float* coeff3, const float* coeff4, cufftComplex* summedSignals, float* mus, float* mus2, float* mus3);

#endif