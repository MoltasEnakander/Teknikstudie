#ifndef BEAMFORMER_WIDEBAND_H
#define BEAMFORMER_WIDEBAND_H
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

#define SAMPLE_RATE (44100.0)       // How many audio samples to capture every second (44100 Hz is standard)
#define NUM_CHANNELS (16)           // Number of audio channels to capture
#define NUM_SECONDS (15)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (10)
#define NUM_BEAMS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define MAX_THREADS_PER_BLOCK (1024)

#define NUM_TAPS (49)
#define NUM_FILTERS (6)
#define F_C (500)
#define BANDWIDTH (2 * F_C * 2 / SAMPLE_RATE)
#define BLOCK_LEN (2048)
#define TEMP (128)

#define FFT_OUTPUT_SIZE (BLOCK_LEN)

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    int maxFrameIndex;              // how many frames that should be processed
    int frameIndex;                 // current number of frames that have been processed
    int numBlocks;
    dim3 threadsPerBlock;

    float* beams;                   // magnitude of the beams
    float* gpu_beams;
    int thetaID[NUM_FILTERS];       // theta index of the strongest beam per frequency block
    int phiID[NUM_FILTERS];         // phi index of the strongest beam per frequency block

    int* a;                         // interpolation data
    int* b;
    float* alpha;
    float* beta;

    fftwf_complex* temp;            // temp storage for the part of the new input block which is to be saved for future use
    fftwf_complex* ordbuffer;       // ordered version of buffer, samples are sorted by channel first, sample id second
    fftwf_complex* block;           // data block to be LP-filtered before beamforming
    cufftComplex* gpu_block;        // block(s) of data passed to the gpu for beamforming
    cufftComplex* summedSignals;    // used to sum up the NUM_CHANNEL signals
    
    fftwf_plan forw_plans[NUM_CHANNELS];                // contains plans for calculating fft:s of the data block(s)
    fftwf_plan back_plans[NUM_CHANNELS * NUM_FILTERS];  // contains plans for calculating inverse fft:s of the block(s) after they have beem filtered

    fftwf_complex* fft_data;            // contains the fft-data for the new block [forw_plans(ordbuffer) = fft_data], ordered by channel
    fftwf_complex* filtered_data;       // result of the pointwise multiplication    
    fftwf_complex* LP_filter;           // lowpass filter used before beamforming

    cufftHandle planMany;
    cufftComplex* summedSignals_fft;
    cufftComplex* summedSignals_fft_BP;    
    cufftComplex* BP_filter;            // fft:s of the bandpass filters
} beamformingData;

// positions in the microphone array
static float xa[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float ya[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

float* linspace(int a, int num);

float* calcDelays(float* theta, float* phi);

int* calca(float* delay);

int* calcb(int* a);

float* calcalpha(float* delay, int* b);

float* calcbeta(float* alpha);

#endif