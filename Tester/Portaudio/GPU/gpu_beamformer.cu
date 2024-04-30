#include "gpu_beamformer.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <chrono>
#include <ctime>

__global__
void interpolateChannels(const float* inputBuffer, float* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta)
{
    int id;
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index of this thread
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * FRAMES_PER_HALFBUFFER; // global index of this thread
    for (int k = 0; k < NUM_CHANNELS; ++k)
    {
        id = k + i * NUM_CHANNELS;
        if (l1 < FRAMES_PER_HALFBUFFER)
        {
            if (max(0, -a[id]) == 0 && l1 < FRAMES_PER_HALFBUFFER - a[id]) // a >= 0
                summedSignals[l2] += alpha[id] * inputBuffer[(l1+a[id])*NUM_CHANNELS + k]; // do not write to the a[id] end positions
            else if (max(0, -a[id]) > 0 && l1 >= a[id]) 
                summedSignals[l2] += alpha[id] * inputBuffer[(l1+a[id])*NUM_CHANNELS + k]; // do not write to the first a[id]-1 positions

            if (max(0, -b[id]) == 0 && l1 < FRAMES_PER_HALFBUFFER - b[id]) // b >= 0
                summedSignals[l2] += beta[id] * inputBuffer[(l1+b[id])*NUM_CHANNELS + k]; // do not write to the b[id] end positions
            else if (max(0, -b[id]) > 0 && l1 >= b[id]) 
                summedSignals[l2] += beta[id] * inputBuffer[(l1+b[id])*NUM_CHANNELS + k]; // do not write to the first b[id]-1 positions
        }
    }
}

__global__ 
void beamforming(const float* inputBuffer, float* beams, const int* a, const int* b, const float* alpha, const float* beta, float* summedSignals/*, float* ya, float* za, float* theta, float* phi*/)
{    
    int i = threadIdx.x; // theta idx    

    // interpolate channels
    interpolateChannels<<<(FRAMES_PER_HALFBUFFER+255)/256, 256>>>(inputBuffer, summedSignals, i, a, b, alpha, beta);
    cudaDeviceSynchronize();

    int idx;
    float beamstrength = 0.0f;
    // normalize
    for (int q = 0; q < FRAMES_PER_HALFBUFFER; ++q)
    {
        idx = q + i * FRAMES_PER_HALFBUFFER;
        summedSignals[idx] /= NUM_CHANNELS;
        summedSignals[idx] = summedSignals[idx] * summedSignals[idx] / FRAMES_PER_HALFBUFFER;
        beamstrength += summedSignals[idx];
    }

    beams[i] = 10 * log10(beamstrength);
}

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
    dim3 threadsPerBlock(NUM_VIEWS * NUM_VIEWS);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    beamforming<<<numBlocks, threadsPerBlock>>>(data->buffer, data->gpubeams, data->a, data->b, data->alpha, data->beta, data->summedSignals/*, data->ya, data->za, data->theta, data->phi*/);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    cudaMemcpy(data->cpubeams, data->gpubeams, NUM_VIEWS*NUM_VIEWS*sizeof(float), cudaMemcpyDeviceToHost);

    int maxID = 0;
    float maxVal = data->cpubeams[0];

    for (int i = 1; i < NUM_VIEWS * NUM_VIEWS; i++)
    {
        if (maxVal < data->cpubeams[i]){
            maxID = i;
            maxVal = data->cpubeams[i];
        }        
    }

    // convert 1d index to 2d index
    data->thetaID = maxID % int(NUM_VIEWS);
    data->phiID = maxID / int(NUM_VIEWS);

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
    cudaMalloc(&(data->a), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->b), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->alpha), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->beta), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->summedSignals), sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES_PER_HALFBUFFER);    
    
    cudaMemcpy(data->a, a, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->b, b, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->alpha, alpha, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->beta, beta, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    
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

    FILE* signal = popen("gnuplot", "w");    

    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        //Pa_Sleep(100);
        // plot maximum direction
        plt::figure(1);
        plt::title("Max direction plot");
        plt::clf();
        plt::scatter(std::vector<float>{theta[data->thetaID] * 180.0f / (float)M_PI}, std::vector<float>{phi[data->phiID] * 180.0f / (float)M_PI}, 25.0, {{"color", "red"}});
        plt::xlim(MIN_VIEW, MAX_VIEW);
        plt::ylim(MIN_VIEW, MAX_VIEW);
        plt::xlabel("theta");
        plt::ylabel("phi");
        plt::grid(true);
        plt::pause(0.15);
        //printf("theta = %f\n", data->theta );
        //printf("phi = %f\n", data->phi );
        //printf("maxframeindex = %d\n", data->maxFrameIndex );
        //printf("frameindex = %d\n", data->frameIndex );
        //fflush(stdout);

        // plot beamforming results in color map
        fprintf(signal, "unset key\n");
        fprintf(signal, "set pm3d\n");
        fprintf(signal, "set view map\n");
        fprintf(signal, "set xrange [ -0.5 : %f ] \n", NUM_VIEWS-0.5);
        fprintf(signal, "set yrange [ -0.5 : %f ] \n", NUM_VIEWS-0.5);
        fprintf(signal, "plot '-' matrix with image\n");
        
        for(int i = 0; i < NUM_VIEWS * NUM_VIEWS; ++i)    
        {
            fprintf(signal, "%f ", data->cpubeams[i]);
            if ((i+1) % NUM_VIEWS == 0)
                fprintf(signal, "\n");
        }
        
        fprintf(signal, "e\n");
        fprintf(signal, "e\n");
        fflush(signal);    

        // Display the buffered changes to stdout in the terminal
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

    // free allocated memory
    cudaFree(data->buffer);
    cudaFree(data->gpubeams);    
    cudaFree(data->a);
    cudaFree(data->b);
    cudaFree(data->alpha);
    cudaFree(data->beta);
    free(delay);
    free(theta);
    free(phi);    
    free(a);
    free(b);
    free(alpha);
    free(beta);
    free(data->cpubeams);
    free(data);

    printf("\n");    

    return EXIT_SUCCESS;
}