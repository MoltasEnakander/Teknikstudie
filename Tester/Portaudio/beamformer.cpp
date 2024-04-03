#include "beamformer.h"

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

    // userData is NULL
    (void)userData;

    // beamform
    std::vector<double> beams = beamforming(in, theta, phi);

    // find strongest beam
    std::vector<double>::iterator result = std::max_element(std::begin(beams), std::end(beams));

    // convert 1d index to 2d index    
    int thetaID = (int)*result / NUM_VIEWS;
    int phiID = (int)*result % NUM_VIEWS;

    // more precise beamforming    
    for (int i = 0; i < NUM_VIEWS; ++i)
    {
        precisetheta[i] = theta[thetaID] + theta[i] / 10;
        precisephi[i] = phi[phiID] + phi[i] / 10;
    }
    beams = beamforming(in, precisetheta, precisephi);

    return 0;
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
        NULL
    );
    checkErr(err);

    // Begin capturing audio
    err = Pa_StartStream(stream);
    checkErr(err);

    // Wait 30 seconds (PortAudio will continue to capture audio)
    Pa_Sleep(30 * 1000);

    // Stop capturing audio
    err = Pa_StopStream(stream);
    checkErr(err);

    // Close the PortAudio stream
    err = Pa_CloseStream(stream);
    checkErr(err);

    // Terminate PortAudio
    err = Pa_Terminate();
    checkErr(err);

    //free(data);

    printf("\n");

    return EXIT_SUCCESS;
}

// OH LORD this feels wrong
std::vector<double> beamforming(const float* inputBuffer, const std::vector<double>& theta, const std::vector<double>& phi)
{    
    int a, b;
    double alpha, beta;
    std::vector<double> beams(NUM_VIEWS*NUM_VIEWS, 0.0);
    std::vector<double> delay(NUM_CHANNELS, 0.0);
    std::vector<double> summedSignal(FRAMES_PER_BUFFER, 0.0); 
    double beamStrength = 0;

    for (int i = 0; i < NUM_VIEWS; ++i) // loop theta directions
    {
        for (int j = 0; j < NUM_VIEWS; ++j) // loop phi directions
        {
            for (int k = 0; k < NUM_CHANNELS; ++k) // loop channels
            {
                delay[k] = -(ya[k] * sind(theta[i]) * cosd(phi[j]) + za[k] * sind(phi[j])) * ARRAY_DIST / C * SAMPLE_RATE;

                // whole samples and fractions of samples
                a = std::floor(delay[k]);
                b = a + 1;
                alpha = b - delay[k];
                beta = 1 - alpha;

                // intepolation of left sample
                for (int l = std::max(1-a, 1); l < std::min(FRAMES_PER_BUFFER-a, FRAMES_PER_BUFFER); ++l)
                {
                    for (int m = 0; m < FRAMES_PER_BUFFER; ++m)
                    {
                        summedSignal[m] += alpha * inputBuffer[l+a + k]; 
                    }
                }

                // interpolation of right sample
                for (int l = std::max(1-b, 1); l < std::min(FRAMES_PER_BUFFER-b, FRAMES_PER_BUFFER); ++l)
                {   
                    for (int m = 0; m < FRAMES_PER_BUFFER; ++m)
                    {
                        summedSignal[m] += beta * inputBuffer[l+b + k]; 
                    }                
                }
            }

            for (int k = 0; k < FRAMES_PER_BUFFER; ++k)
            {
                summedSignal[k] /= 16;
                beamStrength += summedSignal[k];
            }

            beams[i + j*NUM_VIEWS] = 10 * std::log10(std::pow(beamStrength/FRAMES_PER_BUFFER, 2));
        }

        //for (int j = 0; j < NUM_CHANNELS; ++j)
        //{
            //delay[j] = -(ya[j] * sind(theta[i]) * cosd(phi[i])) + za[j] * sind(phi[i]) * ARRAY_DIST / C * SAMPLE_RATE;
        
            // whole samples and fractions of samples
            /*a = std::floor(delay[j]);
            b = a + 1;
            alpha = b - delay[j];
            beta = 1 - alpha;*/

            /*// intepolation of left sample
            for (int k = std::max(1-a, 1); k < std::min(FRAMES_PER_BUFFER-a, FRAMES_PER_BUFFER); ++k)
            {
                for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
                {
                    summedSignal[l] += alpha * inputBuffer[k+a + j]; 
                }
            }

            // interpolation of right sample
            for (int k = std::max(1-b, 1); k < std::min(FRAMES_PER_BUFFER-b, FRAMES_PER_BUFFER); ++k)
            {   
                for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
                {
                    summedSignal[l] += beta * inputBuffer[k+b + j]; 
                }                
            }*/
        //}

        /*for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
        {
            summedSignal[l] /= 16;
            beamStrength += summedSignal[l];
        }

        beams[i] = 10 * std::log10(std::pow(beamStrength/FRAMES_PER_BUFFER, 2));*/

    }

    return beams;
}