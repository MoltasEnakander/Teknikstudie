#include "beamformer.h"

#include <iostream>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <chrono>
#include <ctime>

// Callback data, persisted between calls. Allows us to access the data it
// contains from within the callback function.
//static paTestData* data;

// Checks the return value of a PortAudio function. Logs the message and exits
// if there was an error
static void checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

// PortAudio stream callback function. Will be called after every
// `FRAMES_PER_BUFFER` audio samples PortAudio captures. Used to process the
// resulting audio sample.
static int streamCallback(
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
) {    
    // Cast our input buffer to a float pointer (since our sample format is `paFloat32`)
    float* in = (float*)inputBuffer;

    // We will not be modifying the output buffer. This line is a no-op.
    (void)outputBuffer;

    paTestData* data = (paTestData*)userData;

    long framesToCalc;    
    int finished;
    unsigned long framesLeft = data->maxFrameIndex - data->frameIndex;

    if( framesLeft < framesPerBuffer )
    {
        framesToCalc = framesLeft;
        finished = paComplete;
    }
    else
    {
        framesToCalc = framesPerBuffer;
        finished = paContinue;
    }

    data->frameIndex += framesToCalc;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // beamform
    beamforming(in, theta, phi);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    // find strongest beam    
    auto it = max_element(beams.begin(), beams.end());
    int index = -1;
    if (it != beams.end())  
    {   
        index = it - beams.begin();         
    }
    // index should always be set to something valid by now
    
    // convert 1d index to 2d index
    int thetaID = index % int(NUM_VIEWS);
    int phiID = index / int(NUM_VIEWS);

    //printf("theta: %f\n", theta[thetaID]);
    //printf("phi: %f\n", phi[phiID]);

    data->theta = theta[thetaID];
    data->phi = phi[phiID];

    // more precise beamforming    
    /*for (int i = 0; i < NUM_VIEWS; ++i)
    {
        precisetheta[i] = theta[thetaID] + theta[i] / 10;
        precisephi[i] = phi[phiID] + phi[i] / 10;
    }
    beams = beamforming(in, precisetheta, precisephi);*/

    // Display the buffered changes to stdout in the terminal
    fflush(stdout);

    return finished;
}

int main() 
{
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    // --------------------------------------------------------------------------------------------------------------
    // ------------------------ List all available audio devices and look for desired device ------------------------
    // --------------------------------------------------------------------------------------------------------------
    int numDevices = Pa_GetDeviceCount();
    printf("Number of devices: %d\n", numDevices);

    if (numDevices < 0){
        printf("Error getting device count.\n");
        Pa_Terminate();        
        exit(EXIT_FAILURE);
    }
    else if (numDevices == 0){
        printf("There are no available audio devices on this machine.\n");
        Pa_Terminate();        
        exit(EXIT_FAILURE);
    }

    int device = -1;
    const PaDeviceInfo* deviceInfo;
    for (int i = 0; i < numDevices; i++)
    {
        deviceInfo = Pa_GetDeviceInfo(i);
        printf("Device %d:\n", i);
        printf("    name: %s\n", deviceInfo->name);
        printf("    maxInputChannels: %d\n", deviceInfo->maxInputChannels);
        printf("    maxOutputChannels: %d\n", deviceInfo->maxOutputChannels);
        printf("    defaultSampleRate: %f\n", deviceInfo->defaultSampleRate);

        if (strcmp(deviceInfo->name, DEVICE_NAME) == 0)
        {
            device = i;
        }
    }

    if (device == -1){
        printf("\nDevice: %s not found!\n", DEVICE_NAME);
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }

    printf("Device = %d\n", device);
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    // Define stream capture specifications
    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.device = device;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

    paTestData* data = (paTestData*)malloc(sizeof(paTestData));
    data->maxFrameIndex = NUM_SECONDS * SAMPLE_RATE; // Record for a few seconds.
    data->frameIndex = 0;

    // Open the PortAudio stream
    PaStream* stream;
    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paNoFlag,
        streamCallback,
        data
    );
    checkErr(err);

    // Begin capturing audio
    err = Pa_StartStream(stream);
    checkErr(err);

    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        Pa_Sleep(100);
        plt::clf();
        plt::scatter(std::vector<double>{data->theta}, std::vector<double>{data->phi}, 25.0, {{"color", "red"}});
        plt::xlim(MIN_VIEW, MAX_VIEW);
        plt::ylim(MIN_VIEW, MAX_VIEW);
        plt::xlabel("theta");
        plt::xlabel("phi");
        plt::grid(true);
        plt::pause(0.05);
        //printf("theta = %f\n", data->theta );
        //printf("phi = %f\n", data->phi );
        printf("maxframeindex = %d\n", data->maxFrameIndex );
        printf("frameindex = %d\n", data->frameIndex );
        fflush(stdout);
    }

    // Stop capturing audio
    err = Pa_StopStream(stream);
    checkErr(err);

    // Close the PortAudio stream
    err = Pa_CloseStream(stream);
    checkErr(err);

    // Terminate PortAudio
    err = Pa_Terminate();
    checkErr(err);

    free(data);

    printf("\n");    

    return EXIT_SUCCESS;
}

void beamforming(const float* inputBuffer, const std::vector<double>& theta, const std::vector<double>& phi)
{
    /*
        IMPORTANT NOTE ABOUT inputBuffer:
        index 0 will be the first sample of channel 1
        index 1 will be the first sample of channel 2
        and so on ...
        index NUM_CHANNELS will be the second sample of channel 1
        index NUM_CHANNELS+1 will be the second sample of channel 2
    */
    int a, b, i, j, k, l;
    double alpha, beta, beamStrength;    

    for (i = 0; i < NUM_VIEWS; ++i) // loop theta directions
    {        
        for (j = 0; j < NUM_VIEWS; ++j) // loop phi directions
        {
            std::vector<double> summedSignals(FRAMES_PER_BUFFER, 0.0);
            beamStrength = 0;
            for (k = 0; k < NUM_CHANNELS; ++k) // loop channels
            {
                
                delay[k] = -(ya[k] * sind(theta[i]) * cosd(phi[j]) + za[k] * sind(phi[j])) * ARRAY_DIST / C * SAMPLE_RATE;

                // whole samples and fractions of samples
                a = std::floor(delay[k]);
                b = a + 1;
                alpha = b - delay[k];
                beta = 1 - alpha;

                // interpolation of left sample
                for (l = std::max(-a, 0); l < std::min(FRAMES_PER_BUFFER-a, FRAMES_PER_BUFFER); l++)
                {
                    summedSignals[l] += alpha * inputBuffer[(l+a)*NUM_CHANNELS + k];                    
                }

                // interpolation of right sample
                for (l = std::max(-b, 0); l < std::min(FRAMES_PER_BUFFER-b, FRAMES_PER_BUFFER); l++)
                {   
                    summedSignals[l] += beta * inputBuffer[(l+b)*NUM_CHANNELS + k];
                }
            }
            
            // normalize and calculate "strength" of beam
            for (k = 0; k < FRAMES_PER_BUFFER; k++)
            {
                summedSignals[k] /= NUM_CHANNELS;
                summedSignals[k] = summedSignals[k] * summedSignals[k] / FRAMES_PER_BUFFER;
                beamStrength += summedSignals[k]; 
            }

            beams[i + j*NUM_VIEWS] = 10 * std::log10(beamStrength);     
        }
    }    
}