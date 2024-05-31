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
#include "AudioFile.h"
#include <fftw3.h>

#include <include/pybind11/pybind11.h>
#include <include/pybind11/embed.h>  // python interpreter
#include <include/pybind11/stl.h>  // type conversion

namespace py = pybind11;

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define SAMPLE_RATE (44100.0)       // How many audio samples to capture every second (44100 Hz is standard)
#define NUM_CHANNELS (16)           // Number of audio channels to capture
#define NUM_SECONDS (10)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (10)
#define NUM_VIEWS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define MAX_THREADS_PER_BLOCK (1024)

#define NUM_TAPS (49)
#define NUM_FILTERS (6)
#define F_C (500)
#define BANDWIDTH (2 * F_C * 2 / SAMPLE_RATE)
#define BLOCK_LEN (2048)                                // how long a block will be to store zero padded signals
#define FRAMES_PER_BUFFER (BLOCK_LEN - NUM_TAPS + 1)    // how many samples to save before callback function is called
#define OLA_LENGTH (FRAMES_PER_BUFFER * (NUM_OLA_BLOCK - 1) + BLOCK_LEN)

#define NUM_OLA_BLOCK (8) // how many blocks to store using overlap-add method before starting to apply FFT:s 
#define PADDED_OLA_LENGTH (NUM_OLA_BLOCK * BLOCK_LEN)

#define FFT_OUTPUT_SIZE (BLOCK_LEN)
#define OLA_FFT_OUTPUT_SIZE (PADDED_OLA_LENGTH)

#define DECIMATED_STEP (int(SAMPLE_RATE / (2 * 2 * F_C)))
#define DECIMATED_LENGTH (int(PADDED_OLA_LENGTH / DECIMATED_STEP))

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    int maxFrameIndex;              // how many frames that should be processed
    int frameIndex;                 // current number of frames that have been processed

    float* beams;                   // magnitude of the beams
    int thetaID[NUM_FILTERS];       // theta index of the strongest beam per frequency block
    int phiID[NUM_FILTERS];         // phi index of the strongest beam per frequency block

    fftwf_complex* ordbuffer;       // ordered version of buffer, samples are sorted by channel first, sample id second
    fftwf_complex* summedSignals;        // used to sum up the NUM_CHANNEL signals
    fftwf_plan forw_plans[NUM_CHANNELS]; // contains plans for calculating fft:s for the new incoming block  
    fftwf_plan back_plans[NUM_CHANNELS * NUM_FILTERS]; // contains plans for calculating inverse fft:s of the block after it has beem filtered

    fftwf_complex* fft_data;        // contains the fft-data for the new block [forw_plans(ordbuffer) = fft_data], ordered by channel
    fftwf_complex* firfiltersfft;   // fft of the filters
    fftwf_complex* filtered_data;   // result of the pointwise multiplication
    fftwf_complex* filtered_data_temp;   // temporary container for the filtered data in the time domain [back_plans(filtered_data = filtered_data_temp)], results will be added to OLA_signal
    fftwf_complex* OLA_signal;           // contains the combined signal for each channel and filter, after construction using overlap-add
    
    fftwf_plan OLA_forw[NUM_CHANNELS * NUM_FILTERS]; // plan for calculating the fft of the OLA_signal
    fftwf_plan OLA_back[NUM_CHANNELS * NUM_FILTERS]; // plan for calculating the inverse fft of the OLA_signal after it has been LP-filtered after the IQ-downconversion
    
    int* sine_cosine_counter;       // stores counters used to sync the multiplication of the new OLA_block with the sine and cosine terms                
    fftwf_complex* OLA_fft;         // stores the fft of the OLA_signal
    fftwf_complex* LP_filter;       // lowpass filter used after IQ-downconversion
    
    fftwf_complex* phase_shifts;    // contains the complex exponentials corresponding to phase shifts for each channel, filter and sample
} beamformingData;

// positions in the microphone array
static float ya[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float za[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

float* linspace(int a, int num);

float* calcDelays(float* theta, float* phi);

fftwf_complex* calcPhaseShifts(float* delay);

void listen_live();

void listen_prerecorded(std::vector<AudioFile<float>>& files);

#endif