#ifndef BEAMFORMING_H
#define BEAMFORMER_H

#include "definitions.h"
#include <cufft.h>

__global__
void interpolateChannels(const cufftComplex* inputBuffer, cufftComplex* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta);

__global__ 
void beamforming(const cufftComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftComplex* summedSignals);

#endif