#ifndef GPUBEAMFORMER_H
#define GPUBEAMFORMER_H
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <portaudio.h> // PortAudio: Used for audio capture
#include "AudioFile.h"

#define SAMPLE_RATE (44100.0)   // How many audio samples to capture every second (44100 Hz is standard)
#define FRAMES_PER_BUFFER (2048) // Half of how many audio samples to send to our callback function for each channel
#define NUM_CHANNELS (16)        // Number of audio channels to capture
#define NUM_SECONDS (10)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (5)
#define NUM_VIEWS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define MAX_THREADS_PER_BLOCK (1024)

#define sind(x) (sin(fmod((x),360) * M_PI / 180))
#define cosd(x) (cos(fmod((x),360) * M_PI / 180))

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    int maxFrameIndex;
    int frameIndex;
    float* buffer;
    float* gpubeams;
    float* cpubeams;    
    int* a;
    int* b;
    float* alpha;
    float* beta;    
    int thetaID;
    int phiID;
    float* summedSignals;
} paTestData;

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