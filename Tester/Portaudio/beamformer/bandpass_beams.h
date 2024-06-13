#ifndef BANDPASS_BEAMS_H
#define BANDPASS_BEAMS_H

#include "definitions.h"
#include <cufft.h>
#include <cmath>

__global__
void bandpass_filtering_calcs(int i, cufftDoubleComplex* summedSignals_fft_BP, cufftDoubleComplex* summedSignals_fft, cufftDoubleComplex* BP_filter);

__global__
void bandpass_filtering(cufftDoubleComplex* summedSignals_fft_BP, cufftDoubleComplex* summedSignals_fft, cufftDoubleComplex* BP_filter, double* beams);

#endif