//#include <SFML/Graphics.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <portaudio.h> // PortAudio: Used for audio capture

#define SAMPLE_RATE (44100.0)   // How many audio samples to capture every second (44100 Hz is standard)
#define FRAMES_PER_BUFFER (2048) // How many audio samples to send to our callback function for each channel
#define NUM_CHANNELS (16)        // Number of audio channels to capture
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60)
#define MAX_VIEW (60)
#define VIEW_INTERVAL (10)
#define NUM_VIEWS ((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define sind(x) (sin(fmod((x),360) * M_PI / 180))
#define cosd(x) (cos(fmod((x),360) * M_PI / 180))

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

#define WINDOW_WIDTH (640)
#define WINDOW_HEIGHT (480)

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    //double* in;      // Input buffer, will contain our audio sample
    //double* out;     // Output buffer, FFTW will write to this based on the input buffer's contents    
    //FILE* signal;
    float* buffer;
    float* gpubeams;
    float* cpubeams;
    float* theta;
    float* phi;
    float* ya;
    float* za;
} paTestData;

// positions in the microphone array
static float ya[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float za[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

//void beamforming(const float* inputBuffer, const std::vector<double>& theta, const std::vector<double>& phi);


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

//static std::vector<double> theta = linspace(MIN_VIEW, NUM_VIEWS);
//static std::vector<double> phi = linspace(MIN_VIEW, NUM_VIEWS);

static float* theta = linspace(MIN_VIEW, NUM_VIEWS);
static float* phi = linspace(MIN_VIEW, NUM_VIEWS);

//static std::vector<double> precisetheta(NUM_VIEWS, 0.0);
//static std::vector<double> precisephi(NUM_VIEWS, 0.0);

static std::vector<double> beams((NUM_VIEWS)*(NUM_VIEWS), 4.0);
static std::vector<double> delay(NUM_CHANNELS, 0.0);