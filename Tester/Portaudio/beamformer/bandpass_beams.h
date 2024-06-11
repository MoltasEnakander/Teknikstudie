#ifndef BANDPASS_BEAMS_H
#define BANDPASS_BEAMS_H

#include "definitions.h"
#include <cufft.h>
#include <cmath>

__global__
void bandpass_filtering_calcs(int i, cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter);

__global__
void bandpass_filtering(cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter, float* beams);

#endif