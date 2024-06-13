#ifndef BEAMFORMER_H
#define BEAMFORMER_H
#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>

#include <portaudio.h> // PortAudio: Used for audio capture
#include <fftw3.h>
#include <cufft.h>

#include <include/pybind11/pybind11.h>
#include <include/pybind11/embed.h>  // python interpreter
#include <include/pybind11/stl.h>  // type conversion

namespace py = pybind11;

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include "definitions.h"
#include "beamforming.h"
#include "bandpass_beams.h"
#include "Coeff_Calc.h"
#include "Interpolation_Preparation.h"

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    int maxFrameIndex;              // how many frames that should be processed
    int frameIndex;                 // current number of frames that have been processed
    int numBlocks;
    dim3 threadsPerBlock;

    double* beams;                   // magnitude of the beams
    double* gpu_beams;
    int thetaID[NUM_FILTERS];       // theta index of the strongest beam per frequency block
    int phiID[NUM_FILTERS];         // phi index of the strongest beam per frequency block

    /*int* a;                         // linear interpolation data
    int* b;
    double* alpha;
    double* beta;*/

    fftw_complex* temp;            // temp storage for the part of the new input block which is to be saved for future use
    fftw_complex* ordbuffer;       // ordered version of buffer, samples are sorted by channel first, sample id second
    fftw_complex* block;           // data block to be LP-filtered before beamforming
    cufftDoubleComplex* gpu_block;        // block(s) of data passed to the gpu for beamforming
    cufftDoubleComplex* summedSignals;    // used to sum up the NUM_CHANNEL signals
    
    fftw_plan forw_plans[NUM_CHANNELS];                // contains plans for calculating fft:s of the data block(s)
    fftw_plan back_plans[NUM_CHANNELS * NUM_FILTERS];  // contains plans for calculating inverse fft:s of the block(s) after they have beem filtered

    fftw_complex* fft_data;            // contains the fft-data for the new block [forw_plans(ordbuffer) = fft_data], ordered by channel
    fftw_complex* filtered_data;       // result of the pointwise multiplication    
    fftw_complex* LP_filter;           // lowpass filter used before beamforming

    cufftHandle planMany;
    cufftDoubleComplex* summedSignals_fft;
    cufftDoubleComplex* summedSignals_fft_BP;    
    cufftDoubleComplex* BP_filter;            // fft:s of the bandpass filters

    double* h_coeff1;
    double* h_coeff2;
    double* h_coeff3;
    double* h_coeff4;

    double* coeff1;          // for cubic spline interpolation
    double* coeff2;
    double* coeff3;
    double* coeff4;
    double* cp;
    double* dp;
    double* d;
    double* sigma;

    double* delays;

    double* coefftemp;
    double* coefftemp2;
    double* mus2;
    double* mus3;
    double* mus;

    fftw_complex* ss;

} beamformingData;

#endif