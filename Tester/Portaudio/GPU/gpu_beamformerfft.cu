#include "gpu_beamformerfft.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <chrono>
#include <ctime>
#include <unistd.h>

__global__
void interpolateChannels(const float* inputBuffer, float* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta)
{
    int id;
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index of this thread
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * FRAMES_PER_BUFFER; // global index of this thread
    for (int k = 0; k < NUM_CHANNELS; ++k)
    {
        id = k + i * NUM_CHANNELS;        
        if (max(0, -a[id]) == 0 && l1 < FRAMES_PER_BUFFER - a[id]) // a >= 0
            summedSignals[l2] += alpha[id] * inputBuffer[(l1+a[id])*NUM_CHANNELS + k]; // do not write to the a[id] end positions
        else if (max(0, -a[id]) > 0 && l1 >= a[id]) 
            summedSignals[l2] += alpha[id] * inputBuffer[(l1+a[id])*NUM_CHANNELS + k]; // do not write to the first a[id]-1 positions

        if (max(0, -b[id]) == 0 && l1 < FRAMES_PER_BUFFER - b[id]) // b >= 0
            summedSignals[l2] += beta[id] * inputBuffer[(l1+b[id])*NUM_CHANNELS + k]; // do not write to the b[id] end positions
        else if (max(0, -b[id]) > 0 && l1 >= b[id]) 
            summedSignals[l2] += beta[id] * inputBuffer[(l1+b[id])*NUM_CHANNELS + k]; // do not write to the first b[id]-1 positions        
    }
}

__global__ 
void beamforming(const float* inputBuffer, float* beams, const int* a, const int* b, const float* alpha, const float* beta, float* summedSignals)
{    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_VIEWS * NUM_VIEWS){
        return;
    }

    // interpolate channels    
    interpolateChannels<<<(FRAMES_PER_BUFFER+255)/256, 256>>>(inputBuffer, summedSignals, i, a, b, alpha, beta);
    cudaDeviceSynchronize();

    int idx;
    float beamstrength = 0.0f;
    // normalize
    for (int q = 0; q < FRAMES_PER_BUFFER; ++q)
    {
        idx = q + i * FRAMES_PER_BUFFER;
        summedSignals[idx] /= NUM_CHANNELS;
        summedSignals[idx] = summedSignals[idx] * summedSignals[idx] / FRAMES_PER_BUFFER;
        beamstrength += summedSignals[idx];
    }

    beams[i] = 10 * log10(beamstrength);
}

void free_resources(beamformingData* data){
    // free allocated memory
    cudaFree(data->buffer);
    cudaFree(data->summedSignals);
    cudaFree(data->gpubeams);    
    cudaFree(data->a);
    cudaFree(data->b);
    cudaFree(data->alpha);
    cudaFree(data->beta);    
    cudaFree(data->d_fft_data);    
    free(data->cpubeams);
    free(data->ordbuffer);
    free(data->h_fft_data);    

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_destroy_plan(data->plans[i]);
    }

    free(data);
}

// Checks the return value of a PortAudio function. Logs the message and exits
// if there was an error
static void checkErr(PaError err, beamformingData* data) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        free_resources(data);
        exit(EXIT_FAILURE);
    }
}

// PortAudio stream callback function. Will be called after every
// `2*FRAMES_PER_BUFFER` audio samples PortAudio captures. Used to process the
// resulting audio sample.
static int streamCallback(
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, // framesPerBuffer = 2 * FRAMES_PER_BUFFER
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
) {
    // Cast our input buffer to a float pointer (since our sample format is `paFloat32`)
    float* in = (float*)inputBuffer;

    // We will not be modifying the output buffer. This line is a no-op.
    (void)outputBuffer;

    beamformingData* data = (beamformingData*)userData;
    
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

    cudaMemcpy(data->buffer, in, FRAMES_PER_BUFFER*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice); // copy buffer to GPU memory    
    
    for (int i = 0; i < NUM_CHANNELS; ++i) // sort the incoming buffer based on channel
    {       
        for (int j = 0; j < FRAMES_PER_BUFFER; ++j)
        {           
            data->ordbuffer[i * FRAMES_PER_BUFFER + j] = in[j * NUM_CHANNELS + i];
        }
        
    }

    for (int i = 0; i < NUM_CHANNELS; ++i) // calculate fft for each channel
    {
        fftwf_execute(data->plans[i]);
    }

    cudaMemcpy(data->d_fft_data, data->h_fft_data, SINGLE_OUTPUT_SIZE*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice); // copy buffer to GPU memory

    // beamform
    /*int numBlocks;
    dim3 threadsPerBlock;
    if (NUM_VIEWS * NUM_VIEWS > MAX_THREADS_PER_BLOCK){
        numBlocks = (NUM_VIEWS * NUM_VIEWS) % MAX_THREADS_PER_BLOCK + 1;
        threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK);
    }
    else{
        numBlocks = 1;
        threadsPerBlock = dim3(NUM_VIEWS * NUM_VIEWS);
    }
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    beamforming<<<numBlocks, threadsPerBlock>>>(data->buffer, data->gpubeams, data->a, data->b, data->alpha, data->beta, data->summedSignals);
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
    data->phiID = maxID / int(NUM_VIEWS);*/

    return finished;
}

void listen_live() 
{
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    checkErr(err, nullptr);

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

    // setup interpolation data for the views and channels
    theta = linspace(MIN_VIEW, NUM_VIEWS);
    phi = linspace(MIN_VIEW, NUM_VIEWS);
    delay = calcDelays();
    a = calca();
    b = calcb();
    alpha = calcalpha();
    beta = calcbeta();

    // Define stream capture specifications
    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.device = device;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

    beamformingData* data = (beamformingData*)malloc(sizeof(beamformingData));
    data->maxFrameIndex = NUM_SECONDS * SAMPLE_RATE; // Record for a few seconds.
    data->frameIndex = 0;    
    
    cudaMalloc(&(data->buffer), sizeof(float) * FRAMES_PER_BUFFER * NUM_CHANNELS);    
    cudaMalloc(&(data->gpubeams), sizeof(float) * NUM_VIEWS * NUM_VIEWS);
    cudaMalloc(&(data->a), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->b), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->alpha), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->beta), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->summedSignals), sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES_PER_BUFFER);    
    cudaMalloc(&(data->d_fft_data), sizeof(float) * SINGLE_OUTPUT_SIZE * NUM_CHANNELS);
    
    cudaMemcpy(data->a, a, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->b, b, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->alpha, alpha, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->beta, beta, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);

    free(theta); free(phi); free(delay); free(a); free(b); free(alpha); free(beta); // free memory which does not have to be allocated anymore
    
    data->cpubeams = (float*)malloc(NUM_VIEWS * NUM_VIEWS * sizeof(float));
    data->ordbuffer = (float*)fftwf_malloc(FRAMES_PER_BUFFER * NUM_CHANNELS * sizeof(float));
    data->h_fft_data = (fftwf_complex*)fftwf_malloc(SINGLE_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));

    for (int i = 0; i < NUM_CHANNELS; ++i) // create the plans for calculating the fft of each channel block
    {
        data->plans[i] = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, &data->ordbuffer[i * FRAMES_PER_BUFFER], &data->h_fft_data[i * SINGLE_OUTPUT_SIZE], FFTW_ESTIMATE);
    }

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
    checkErr(err, data);

    // Begin capturing audio
    err = Pa_StartStream(stream);
    checkErr(err, data);

    FILE* signal = popen("gnuplot", "w");    
    //const int nrows = 4, ncols = 4;    
    //int row, col;

    std::vector<int> bins;
    for (int i = 0; i < SINGLE_OUTPUT_SIZE; ++i)
    {
        bins.push_back(i);
    }

    std::vector<float> ch1(SINGLE_OUTPUT_SIZE);
    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        for (int i = 0; i < SINGLE_OUTPUT_SIZE; ++i)
        {
            ch1.at(i) = sqrt(data->h_fft_data[i][0] * data->h_fft_data[i][0] + data->h_fft_data[i][1] * data->h_fft_data[i][1]);
        }

        plt::figure(7);
        plt::title("Frequency contents, channel 1");
        plt::clf();    
        plt::plot(bins, ch1);
        plt::xlabel("freq bin");
        
        plt::pause(0.05);

        //Pa_Sleep(100);
        // plot maximum direction
        /*plt::figure(1);
        plt::title("Max direction plot");
        plt::clf();
        plt::scatter(std::vector<float>{theta[data->thetaID] * 180.0f / (float)M_PI}, std::vector<float>{phi[data->phiID] * 180.0f / (float)M_PI}, 25.0, {{"color", "red"}});
        plt::xlim(MIN_VIEW, MAX_VIEW);
        plt::ylim(MIN_VIEW, MAX_VIEW);
        plt::xlabel("theta");
        plt::ylabel("phi");
        plt::grid(true);
        plt::pause(0.15);*/
        //printf("theta = %f\n", data->theta );
        //printf("phi = %f\n", data->phi );
        //printf("maxframeindex = %d\n", data->maxFrameIndex );
        //printf("frameindex = %d\n", data->frameIndex );
        //fflush(stdout);

        // plot frequency contents of channels
        /*plt::figure(2);
        plt::title("Frequency contents");
        plt::clf();
        for(int w = 0; w < NUM_CHANNELS; ++w){
            row = w / 4;
            col = w % 4;
            plt::subplot2grid(nrows, ncols, row, col);
            plt::plot({1.0,2.0,3.0,4.0});
            plt::xlabel("freq bin");
        }
        //plt::show();
        plt::pause(0.02);
        


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
        fflush(stdout);*/
    }    

    // Stop capturing audio
    err = Pa_StopStream(stream);
    checkErr(err, data);

    // Close the PortAudio stream
    err = Pa_CloseStream(stream);
    checkErr(err, data);

    // Terminate PortAudio
    err = Pa_Terminate();
    checkErr(err, data);

    free_resources(data);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////// UTILITY FUNCTIONS ///////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

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

float* calcDelays()
{
    float* d = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));   

    int pid = 0;
    int tid = 0;
    for (int i = 0; i < NUM_VIEWS * NUM_VIEWS; ++i){
        for (int k = 0; k < NUM_CHANNELS; ++k){
            d[k + i * NUM_CHANNELS] = -(ya[k] * sinf(theta[tid]) * cosf(phi[pid]) + za[k] * sinf(phi[pid])) * ARRAY_DIST / C * SAMPLE_RATE;
        }
        tid++;
        if (tid >= NUM_VIEWS){
            tid = 0;
            pid++;
        }
    }
    return d;
}

int* calca()
{
    int* a = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        a[i] = floor(delay[i]);
    }
    return a;
}

int* calcb()
{
    int* b = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        b[i] = a[i] + 1;
    }
    return b;
}

float* calcalpha()
{
    float* alpha = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        alpha[i] = b[i] - delay[i];
    }
    return alpha;
}

float* calcbeta()
{
    float* beta = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        beta[i] = 1 - alpha[i];
    }
    return beta;
}