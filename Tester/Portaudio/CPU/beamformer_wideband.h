#ifndef BEAMFORMER_WIDEBAND_H
#define BEAMFORMER_WIDEBAND_H
#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <portaudio.h> // PortAudio: Used for audio capture
#include "AudioFile.h"
#include <fftw3.h>

#include <include/pybind11/pybind11.h>
#include <include/pybind11/embed.h>  // python interpreter
#include <include/pybind11/stl.h>  // type conversion

#define SAMPLE_RATE (44100.0)       // How many audio samples to capture every second (44100 Hz is standard)
#define NUM_CHANNELS (16)           // Number of audio channels to capture
#define NUM_SECONDS (10)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (5)
#define NUM_VIEWS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define MAX_THREADS_PER_BLOCK (1024)

#define NUM_TAPS (49)
#define NUM_FILTERS (6)
#define BANDWIDTH (1000 * 2 / SAMPLE_RATE)
#define BLOCK_LEN (2048)                                    // how long a block will be to store zero padded signals
#define FRAMES_PER_BUFFER (FFT_BLOCK_LEN - NUM_TAPS + 1)    // how many samples to save before callback function is called

#define FFT_OUTPUT_SIZE (BLOCK_LEN / 2 + 1)

#define sind(x) (sin(fmod((x),360) * M_PI / 180))
#define cosd(x) (cos(fmod((x),360) * M_PI / 180))

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    int maxFrameIndex;              // how many frames that should be processed
    int frameIndex;                 // current number of frames that have been processed    
    float* ordbuffer;               // ordered version of buffer, samples are sorted by channel first, sample id second
    
    float* beams;                   // magnitude of the beams, stored on the cpu
    int* a;                         // a, b, alpha and beta are used for interpolation
    int* b;
    float* alpha;
    float* beta;    
    int thetaID;                    // theta index of the strongest beam
    int phiID;                      // phi index of the strongest beam
    float* summedSignals;           // contains a combined timesignal after each channel has been interpolated and the signals have been summed together
    fftwf_plan forw_plans[NUM_CHANNELS]; // contains plans for calculating fft:s    
    fftwf_plan back_plans[NUM_CHANNELS]; // contains plans for calculating inverse fft:s

    fftwf_complex* fft_data;        // contains the fft-data for each channel
    fftwf_complex* firfiltersfft;
    
} beamformingData;

// positions in the microphone array
static float ya[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float za[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

float* linspace(int a, int num);
static float* theta = NULL;
static float* phi = NULL;

float* calcDelays();
static float* delay = NULL;

int* calca();
static int* a = NULL;

int* calcb();
static int* b = NULL;

float* calcalpha();
static float* alpha = NULL;

float* calcbeta();
static float* beta = NULL;

void listen_live();

void listen_prerecorded(std::vector<AudioFile<float>>& files);

#endif