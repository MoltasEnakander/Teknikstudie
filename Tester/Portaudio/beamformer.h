#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <portaudio.h> // PortAudio: Used for audio capture
//#include <fftw3.h>     // FFTW:      Provides a discrete FFT algorithm to get frequency data from captured audio

#define SAMPLE_RATE 44100.0   // How many audio samples to capture every second (44100 Hz is standard)
#define FRAMES_PER_BUFFER 2048 // How many audio samples to send to our callback function for each channel
#define NUM_CHANNELS 16        // Number of audio channels to capture
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define SPECTRO_FREQ_START 20  // Lower bound of the displayed spectrogram (Hz)
#define SPECTRO_FREQ_END 20000 // Upper bound of the displayed spectrogram (Hz)

#define MIN_VIEW -60
#define MAX_VIEW 60
#define VIEW_INTERVAL 10
#define NUM_VIEWS (MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1

#define sind(x) (sin(fmod((x),360) * M_PI / 180))
#define cosd(x) (cos(fmod((x),360) * M_PI / 180))

#define C 340 // m/s
#define ARRAY_DIST 0.042 // m

// Define our callback data (data that is passed to every callback function call)
typedef struct {
    //double* in;      // Input buffer, will contain our audio sample
    //double* out;     // Output buffer, FFTW will write to this based on the input buffer's contents
    //fftw_plan p;     // Created by FFTW to facilitate FFT calculation
    //int startIndex;  // First index of our FFT output to display in the spectrogram
    //int spectroSize; // Number of elements in our FFT output to display from the start index
    FILE* signal;    
} paTestData;

// positions in the microphone array
static double ya[16] = {-0.5, -1.5, -0.5, -1.5, -0.5, -1.5, -0.5, -1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5};
static double za[16] = {-1.5, -1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, -0.5, -0.5, -1.5, -1.5};

std::vector<double> beamforming(const float* inputBuffer, const std::vector<double>& theta, const std::vector<double>& phi);

std::vector<double> linspace(int a, int num)
{
    // create a vector of length num
    std::vector<double> v(NUM_VIEWS, 0);
    double tmp = 0.0;         
             
    // now assign the values to the vector
    for (int i = 0; i < num; i++)
    {
        v[i] = a + i * VIEW_INTERVAL;
    }
    return v;
}

static std::vector<double> theta = linspace(MIN_VIEW, NUM_VIEWS);
static std::vector<double> phi = linspace(MIN_VIEW, NUM_VIEWS);
static std::vector<double> precisetheta(NUM_VIEWS, 0.0);
static std::vector<double> precisephi(NUM_VIEWS, 0.0);
