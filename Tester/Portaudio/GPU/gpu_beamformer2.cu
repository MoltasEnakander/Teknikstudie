#include "gpu_beamformer2.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

__global__ 
void beamforming(float* inputBuffer, float* beams, float* theta, float* phi, float* ya, float* za)
{
    // first parallelization: one view per call
    int i = threadIdx.x; // theta idx
    int j = threadIdx.y; // phi idx    

    int a, b, k, l;
    float alpha, beta, beamStrength;
    float delay;
    
    float summedSignals[FRAMES_PER_HALFBUFFER];
    beamStrength = 0;
    for (k = 0; k < NUM_CHANNELS; k++) // loop channels
    {                
        delay = -(ya[k] * sinf(theta[i]) * cosf(phi[j]) + za[k] * sinf(phi[j])) * ARRAY_DIST / C * SAMPLE_RATE;

        // whole samples and fractions of samples
        a = floor(delay);
        b = a + 1;
        alpha = b - delay;
        beta = 1 - alpha;

        // interpolation of left sample
        for (l = max(-a, 0); l < min(FRAMES_PER_HALFBUFFER-a, FRAMES_PER_HALFBUFFER); l++)
        {
            summedSignals[l] += alpha * inputBuffer[(l+a)*NUM_CHANNELS + k];
        }

        // interpolation of right sample
        for (l = max(-b, 0); l < min(FRAMES_PER_HALFBUFFER-b, FRAMES_PER_HALFBUFFER); l++)
        {   
            summedSignals[l] += beta * inputBuffer[(l+b)*NUM_CHANNELS + k];
        }
    }
    
    // normalize and calculate "strength" of beam
    for (k = 0; k < FRAMES_PER_HALFBUFFER; k++)
    {
        summedSignals[k] /= NUM_CHANNELS;
        summedSignals[k] = summedSignals[k] * summedSignals[k] / FRAMES_PER_HALFBUFFER;
        beamStrength += summedSignals[k];
    }

    //beams[i + j * NUM_VIEWS] = i*10.0f + j*1200.0f;
    beams[i + j*NUM_VIEWS] = 10 * std::log10(beamStrength);
}

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
// `2*FRAMES_PER_HALFBUFFER` audio samples PortAudio captures. Used to process the
// resulting audio sample.
static int streamCallback(
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, // framesPerBuffer = 2 * FRAMES_PER_HALFBUFFER
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
) {
    // Cast our input buffer to a float pointer (since our sample format is `paFloat32`)
    float* in = (float*)inputBuffer;

    // We will not be modifying the output buffer. This line is a no-op.
    (void)outputBuffer;

    paTestData* data = (paTestData*)userData;
    
    int finished;
    unsigned long framesLeft = data->maxFrameIndex - data->frameIndex;

    if( framesLeft < framesPerBuffer )
    {
        data->frameIndex += framesLeft;
        finished = paComplete;
    }
    else
    {
        data->frameIndex += framesPerBuffer;
        finished = paContinue;
    }    

    cudaMemcpy(data->buffer, in, FRAMES_PER_HALFBUFFER*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice); // copy buffer to GPU memory    

    // beamform
    int numBlocks = 1;
    dim3 threadsPerBlock(NUM_VIEWS, NUM_VIEWS);
    beamforming<<<numBlocks, threadsPerBlock>>>(data->buffer, data->gpubeams, data->theta, data->phi, data->ya, data->za);

    cudaMemcpy(data->cpubeams, data->gpubeams, NUM_VIEWS*NUM_VIEWS*sizeof(float), cudaMemcpyDeviceToHost);

    int maxID = 0;
    float maxVal = data->cpubeams[0];

    for (int i = 1; i < NUM_VIEWS * NUM_VIEWS; i++)
    {
        if (maxVal < data->cpubeams[i]){
            maxID = i;
            maxVal = data->cpubeams[i];
        }            
        //printf("Beam %d: %f\n", i, data->cpubeams[i]);
    }
    
    // convert 1d index to 2d index
    data->thetaID = maxID % int(NUM_VIEWS);
    data->phiID = maxID / int(NUM_VIEWS);

    //printf("theta: %f\n", theta[thetaID] * 180.0f / M_PI);
    //printf("phi: %f\n", phi[phiID] * 180.0f / M_PI);

    // Display the buffered changes to stdout in the terminal
    //fflush(stdout);

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

    cudaMalloc(&(data->buffer), sizeof(float) * FRAMES_PER_HALFBUFFER * NUM_CHANNELS);
    cudaMalloc(&(data->gpubeams), sizeof(float) * NUM_VIEWS * NUM_VIEWS);
    cudaMalloc(&(data->theta), sizeof(float) * NUM_VIEWS);
    cudaMalloc(&(data->phi), sizeof(float) * NUM_VIEWS);
    cudaMalloc(&(data->ya), sizeof(float) * NUM_CHANNELS);
    cudaMalloc(&(data->za), sizeof(float) * NUM_CHANNELS);    

    cudaMemcpy(data->theta, theta, NUM_VIEWS*sizeof(float), cudaMemcpyHostToDevice); // copy theta to GPU memory
    cudaMemcpy(data->phi, phi, NUM_VIEWS*sizeof(float), cudaMemcpyHostToDevice); // copy phi to GPU memory
    cudaMemcpy(data->ya, ya, NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice); // copy ya to GPU memory
    cudaMemcpy(data->za, za, NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice); // copy za to GPU memory
    
    data->cpubeams = (float*)malloc(NUM_VIEWS*NUM_VIEWS*sizeof(float));

    // Open the PortAudio stream
    PaStream* stream;
    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL,
        SAMPLE_RATE,
        FRAMES_PER_HALFBUFFER*2,
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
        //Pa_Sleep(100);
        plt::clf();
        plt::scatter(std::vector<float>{theta[data->thetaID] * 180.0f / (float)M_PI}, std::vector<float>{phi[data->phiID] * 180.0f / (float)M_PI}, 25.0, {{"color", "red"}});
        plt::xlim(MIN_VIEW, MAX_VIEW);
        plt::ylim(MIN_VIEW, MAX_VIEW);
        plt::xlabel("theta");
        plt::xlabel("phi");
        plt::grid(true);
        plt::pause(0.15);
        //printf("theta = %f\n", data->theta );
        //printf("phi = %f\n", data->phi );
        //printf("maxframeindex = %d\n", data->maxFrameIndex );
        //printf("frameindex = %d\n", data->frameIndex );
        //fflush(stdout);
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

    cudaFree(data->buffer);
    cudaFree(data->gpubeams);
    cudaFree(data->theta);
    cudaFree(data->phi);
    cudaFree(data->ya);
    cudaFree(data->za);
    free(theta);
    free(phi);
    free(data->cpubeams);
    free(data);

    printf("\n");    

    return EXIT_SUCCESS;
}