#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <portaudio.h> // PortAudio: Used for audio capture

#define SAMPLE_RATE (44100.0)   // How many audio samples to capture every second (44100 Hz is standard)
#define FRAMES_PER_HALFBUFFER (1024) // Half of how many audio samples to send to our callback function for each channel
#define NUM_CHANNELS (16)        // Number of audio channels to capture
#define NUM_SECONDS (10)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (5)
#define NUM_VIEWS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

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
    float* theta;
    float* phi;    
    int thetaID;
    int phiID;
    float* summedSignals;
} paTestData;

// positions in the microphone array
static float ya[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float za[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

float* linspace(int a, int num)
{
    // create a vector of length num
    //std::vector<double> v(NUM_VIEWS, 0);    
    float* f = (float*)malloc(NUM_VIEWS*sizeof(float));
             
    // now assign the values to the vector
    for (int i = 0; i < num; i++)
    {
        f[i] = (a + i * VIEW_INTERVAL) * M_PI / 180.0f;
    }
    return f;
}
static float* theta = linspace(MIN_VIEW, NUM_VIEWS);
static float* phi = linspace(MIN_VIEW, NUM_VIEWS);

float* calcDelays()
{
    float* d = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS; ++i)
    {
        for (int j = 0; j < NUM_VIEWS; ++j)
        {
            for (int k = 0; k < NUM_CHANNELS; ++k)
            {                
                d[k + j*NUM_CHANNELS + i*NUM_CHANNELS*NUM_VIEWS] = -(ya[k] * sinf(theta[i]) * cosf(phi[j]) + za[k] * sinf(phi[j])) * ARRAY_DIST / C * SAMPLE_RATE;
            }
        }
    }
    return d;
}
static float* delay = calcDelays();

int* calca()
{
    int* a = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        a[i] = floor(delay[i]);
    }
    return a;
}
static int* a = calca();

int* calcb()
{
    int* b = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        b[i] = a[i] + 1;
    }
    return b;
}
static int* b = calcb();

float* calcalpha()
{
    float* alpha = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        alpha[i] = b[i] - delay[i];
    }
    return alpha;
}
static float* alpha = calcalpha();

float* calcbeta()
{
    float* beta = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        beta[i] = 1 - alpha[i];
    }
    return beta;
}
static float* beta = calcbeta();