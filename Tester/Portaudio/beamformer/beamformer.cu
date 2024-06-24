#include "beamformer.h"

#include <chrono>
#include <ctime>
#include <unistd.h>

#include <thread>

#include <iostream>
#include <fstream>

void free_resources(beamformingData* data)
{
    // free allocated memory
    free(data->beams);
    cudaFree(data->gpu_beams);
    cudaFree(data->a);
    cudaFree(data->alpha);
    cudaFree(data->b);
    cudaFree(data->beta);
    fftwf_free(data->temp);
    fftwf_free(data->ordbuffer);
    fftwf_free(data->block);
    cudaFree(data->gpu_block)
    cudaFree(data->summedSignals);   
    fftwf_free(data->fft_data);    
    fftwf_free(data->filtered_data);    
    fftwf_free(data->LP_filter);        
    cudaFree(data->summedSignals_fft);
    cudaFree(data->summedSignals_fft_BP);
    cudaFree(data->BP_filter);
    
    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_destroy_plan(data->forw_plans[i]);
        fftwf_destroy_plan(data->back_plans[i]);                
    } 

    cufftDestroy(data->planMany);    
    
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
// BLOCK_LEN audio samples PortAudio captures. Used to process the
// resulting audio sample.
static int streamCallback(
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
)
{
    // Cast our input buffer to a float pointer (since our sample format is `pafloat32`)
    float* in = (float*)inputBuffer;

    beamformingData* data = (beamformingData*)userData;
    
    // keep track of when to stop listening
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

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    for (int i = 0; i < NUM_CHANNELS; ++i) // sort the incoming buffer based on channel
    {       
        for (int j = 0; j < BLOCK_LEN; ++j)
        {            
            data->ordbuffer[i * BLOCK_LEN + j][0] = in[j * NUM_CHANNELS + i];
            //data->ordbuffer[i * BLOCK_LEN + j][0] = in[j];
            data->ordbuffer[i * BLOCK_LEN + j][1] = 0.0f;            
        }        
    }

    for (int i = 0; i < NUM_CHANNELS; ++i) // build data block to be processed
    {
        // 1. move the last part of the old input into the beginning of the block
        // 2. fill the rest of the block with BLOCK_LEN - TEMP values from the new input
        // 3. save the last TEMP values from the new input to the temp storage for use in next call
        std::memcpy(&(data->block[i * BLOCK_LEN]), &(data->temp[i * TEMP]), TEMP * sizeof(fftwf_complex)); 
        std::memcpy(&(data->block[i * BLOCK_LEN + TEMP]), &(data->ordbuffer[i * BLOCK_LEN]), (BLOCK_LEN - TEMP) * sizeof(fftwf_complex));
        std::memcpy(&(data->temp[i * TEMP]), &(data->ordbuffer[i * BLOCK_LEN + (BLOCK_LEN - TEMP)]), TEMP * sizeof(fftwf_complex));
    }

    for (int i = 0; i < NUM_CHANNELS; ++i) // calculate fft for each channel
    {
        fftwf_execute(data->forw_plans[i]);
    }

    // perform lowpass filtering in freq domain
    int resultID, dataID;
    for (int i = 0; i < NUM_CHANNELS; ++i) // for every channel
    {
        for (int j = 0; j < FFT_OUTPUT_SIZE; ++j) // for all samples
        {                
            // j denotes frequency bin            
            // i denotes the channel
            resultID = j + i * FFT_OUTPUT_SIZE;
            dataID = j + i * FFT_OUTPUT_SIZE;
            data->filtered_data[resultID][0] = data->fft_data[dataID][0] * data->LP_filter[j][0] - data->fft_data[dataID][1] * data->LP_filter[j][1];
            data->filtered_data[resultID][1] = data->fft_data[dataID][0] * data->LP_filter[j][1] + data->fft_data[dataID][1] * data->LP_filter[j][0];                
        }
        // inverse fourier transform to get back signals in time domain.        
        fftwf_execute(data->back_plans[i]); // amplitude gain BLOCK_LEN
    }

    // copy data blocks to gpu
    cudaMemcpy(data->gpu_block, data->block, BLOCK_LEN*NUM_CHANNELS*sizeof(fftwf_complex), cudaMemcpyHostToDevice); // copy buffer to GPU memory    

    // create beams    
    beamforming<<<data->numBlocks, data->threadsPerBlock>>>(data->gpu_block, data->a, data->b, data->alpha, data->beta, data->summedSignals);
    cudaDeviceSynchronize();

    cufftExecC2C(data->planMany, data->summedSignals, data->summedSignals_fft, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    bandpass_filtering<<<data->numBlocks, data->threadsPerBlock>>>(data->summedSignals_fft_BP, data->summedSignals_fft, data->BP_filter, data->gpu_beams);
    cudaDeviceSynchronize();    

    // copy the intensity of the beams to the cpu
    cudaMemcpy(data->beams, data->gpu_beams, NUM_BEAMS*NUM_BEAMS*NUM_FILTERS*sizeof(float), cudaMemcpyDeviceToHost);
    
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    return finished;
}

int main() 
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

    // setup necessary data containers for the beamforming
    beamformingData* data = (beamformingData*)malloc(sizeof(beamformingData));
    data->maxFrameIndex = NUM_SECONDS * SAMPLE_RATE; // Record for a few seconds.
    data->frameIndex = 0;

    if (NUM_BEAMS * NUM_BEAMS > MAX_THREADS_PER_BLOCK){
        data->numBlocks = (NUM_BEAMS * NUM_BEAMS) % MAX_THREADS_PER_BLOCK + 1;
        data->threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK);
    }
    else{
        data->numBlocks = 1;
        data->threadsPerBlock = dim3(NUM_BEAMS * NUM_BEAMS);
    }

    printf("Setting up fir filters.\n");    
    py::scoped_interpreter python;

    py::function my_func =
        py::reinterpret_borrow<py::function>(
            py::module::import("beamformer.filtercreation").attr("filtercreation")
    );    
    
    py::list res = my_func(NUM_FILTERS, NUM_TAPS, BANDWIDTH); // create the filters
    // temporary save state of data
    std::vector<float> taps;
    for (py::handle obj : res) {  // iterators!
        taps.push_back(obj.attr("__float__")().cast<float>());
    }

    py::list res2 = my_func(1, NUM_TAPS, 10000.0 / 22050.0);
    // temporary save state of data
    std::vector<float> taps2;
    for (py::handle obj : res2) {  // iterators!
        taps2.push_back(obj.attr("__float__")().cast<float>());
    }

    // transfer data for real, goal is to get a buffer that looks like (with zero-padded signals):
    // filter1[0], filter1[1], ..., 0, 0, 0, filter2[0], filter2[1], ..., 0, 0, 0
    // -------- BLOCK_LEN samples ---------, -------- BLOCK_LEN samples --------- 
    fftwf_complex* firfilters = (fftwf_complex*)malloc(BLOCK_LEN * NUM_FILTERS * sizeof(fftwf_complex));
    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        for (int j = 0; j < BLOCK_LEN; ++j)
        {
            if (j < NUM_TAPS)
                firfilters[i * BLOCK_LEN + j][0] = taps[NUM_TAPS * i + j];
            else
                firfilters[i * BLOCK_LEN + j][0] = 0.0; // zero pad filters
            firfilters[i * BLOCK_LEN + j][1] = 0.0;
        }
    }
    taps.clear();

    fftwf_complex* lpfilter = (fftwf_complex*)malloc(BLOCK_LEN * sizeof(fftwf_complex));
    for (int i = 0; i < BLOCK_LEN; ++i)
    {
        if (i < NUM_TAPS)
            lpfilter[i][0] = taps2[i];
        else
            lpfilter[i][0] = 0.0; // zero pad filters
        lpfilter[i][1] = 0.0;
    }
    taps2.clear();

    // apply fft to filters
    fftwf_complex* firfiltersfft = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_FILTERS * sizeof(fftwf_complex));
    data->LP_filter = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * sizeof(fftwf_complex));
    fftwf_plan filter_plans[NUM_FILTERS];
    fftwf_plan lp_filter_plan;
    for (int i = 0; i < NUM_FILTERS; ++i) // create the plans for calculating the fft of each filter block
    {
        filter_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &firfilters[i * BLOCK_LEN], &firfiltersfft[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE);
    }
    lp_filter_plan = fftwf_plan_dft_1d(BLOCK_LEN, lpfilter, data->LP_filter, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_execute(filter_plans[i]);
    }
    fftwf_execute(lp_filter_plan);
    
    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_destroy_plan(filter_plans[i]);
    }
    fftwf_destroy_plan(lp_filter_plan);

    cudaMalloc(&(data->BP_filter), sizeof(cufftComplex) * BLOCK_LEN * NUM_FILTERS);
    cudaMemcpy(data->BP_filter, firfiltersfft, sizeof(cufftComplex) * BLOCK_LEN * NUM_FILTERS, cudaMemcpyHostToDevice);    

    free(firfilters);
    free(firfiltersfft);
    free(lpfilter);    

    printf("Create interpolation data.\n");
    float* theta = linspace(MIN_VIEW, NUM_BEAMS);
    float* phi = linspace(MIN_VIEW, NUM_BEAMS);
    float* delay = calcDelays(theta, phi);

    int* a = calca(delay);
    int* b = calcb(a);
    float* alpha = calcalpha(delay, b);
    float* beta = calcbeta(alpha);

    cudaMalloc(&(data->a), sizeof(int) * NUM_BEAMS * NUM_BEAMS * NUM_CHANNELS);
    cudaMalloc(&(data->b), sizeof(int) * NUM_BEAMS * NUM_BEAMS * NUM_CHANNELS);
    cudaMalloc(&(data->alpha), sizeof(float) * NUM_BEAMS * NUM_BEAMS * NUM_CHANNELS);
    cudaMalloc(&(data->beta), sizeof(float) * NUM_BEAMS * NUM_BEAMS * NUM_CHANNELS);
    cudaMemcpy(data->a, a, NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->b, b, NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->alpha, alpha, NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->beta, beta, NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);

    free(theta); free(phi); free(delay); free(a); free(b); free(alpha); free(beta); // free memory which does not have to be allocated anymore*/    

    printf("Create remaining buffers\n");
    data->beams = (float*)malloc(NUM_BEAMS * NUM_BEAMS * NUM_FILTERS * sizeof(float));
    std::memset(data->beams, 0.0, NUM_BEAMS * NUM_BEAMS * NUM_FILTERS * sizeof(float));
    cudaMalloc(&(data->gpu_beams), sizeof(float) * NUM_BEAMS * NUM_BEAMS * NUM_FILTERS);

    cudaMalloc(&(data->gpu_block), sizeof(cufftComplex) * NUM_CHANNELS * BLOCK_LEN);

    data->temp = (fftwf_complex*)fftwf_malloc(TEMP * NUM_CHANNELS * sizeof(fftwf_complex));
    for (int i = 0; i < TEMP * NUM_CHANNELS; ++i)
    {
        data->temp[i][0] = 0.0;
        data->temp[i][1] = 0.0;
    }

    data->ordbuffer = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));
    data->block = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));
    
    cudaMalloc(&(data->summedSignals), sizeof(cufftComplex) * NUM_BEAMS * NUM_BEAMS * BLOCK_LEN);
    cudaMalloc(&(data->summedSignals_fft), sizeof(cufftComplex) * NUM_BEAMS * NUM_BEAMS * BLOCK_LEN);
    cudaMalloc(&(data->summedSignals_fft_BP), sizeof(cufftComplex) * NUM_BEAMS * NUM_BEAMS * BLOCK_LEN * NUM_FILTERS);

    data->fft_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));
    data->filtered_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));    

    for (int i = 0; i < BLOCK_LEN * NUM_CHANNELS; ++i)
    {
        data->ordbuffer[i][0] = 0.0;
        data->ordbuffer[i][1] = 0.0;
    }
    
    for (int i = 0; i < BLOCK_LEN * NUM_CHANNELS; ++i)
    {
        data->block[i][0] = 0.0;
        data->block[i][1] = 0.0;
    }

    printf("Creating fft plans.\n");
    int n[1] = {BLOCK_LEN};
    int inembed[] = {BLOCK_LEN};
    int onembed[] = {BLOCK_LEN};
    
    cufftPlanMany(&(data->planMany), 1, n, inembed, 1, BLOCK_LEN, onembed, 1, BLOCK_LEN, CUFFT_C2C, NUM_BEAMS*NUM_BEAMS);

    for (int i = 0; i < NUM_CHANNELS; ++i) // create the plans for calculating the fft of each channel block
    {
        data->forw_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &data->block[i * BLOCK_LEN], &data->fft_data[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE); // NUM_CHANNELS channels for each block which requires FFT_OUTPUT_SIZE spots to store the fft data
        data->back_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &data->filtered_data[i * FFT_OUTPUT_SIZE], &data->block[i * BLOCK_LEN], FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    printf("Defining stream parameters.\n");
    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.device = device;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

    // Open the PortAudio stream
    printf("Starting stream.\n");    
    PaStream* stream;
    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL,
        SAMPLE_RATE,
        BLOCK_LEN,
        paNoFlag,
        streamCallback,
        data
    );
    checkErr(err, data);

    // Begin capturing audio
    err = Pa_StartStream(stream);
    checkErr(err, data);

    FILE* signal = popen("gnuplot", "w");
    //FILE* signal2 = popen("gnuplot", "w");
    //FILE* signal3 = popen("gnuplot", "w");

    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )    
    {
        // plot beamforming results in color map
        fprintf(signal, "unset key\n");
        fprintf(signal, "set pm3d\n");
        fprintf(signal, "set view map\n");
        fprintf(signal, "set xrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal, "set yrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal, "plot '-' matrix with image\n");
        
        for(int i = 3 * NUM_BEAMS * NUM_BEAMS; i < 4 * NUM_BEAMS * NUM_BEAMS; i++) // plot map for the lowest frequency band    
        {
            fprintf(signal, "%f ", data->beams[i]);            
            if ((i+1) % NUM_BEAMS == 0)
                fprintf(signal, "\n");            
        }
        
        fprintf(signal, "\ne\n");

        fflush(signal);

        /*fprintf(signal2, "unset key\n");
        fprintf(signal2, "set pm3d\n");
        fprintf(signal2, "set view map\n");
        fprintf(signal2, "set xrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal2, "set yrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal2, "plot '-' matrix with image\n");
        
        for(int i = 1 * NUM_BEAMS * NUM_BEAMS; i < 2 * NUM_BEAMS * NUM_BEAMS; i++)
        {
            fprintf(signal2, "%f ", data->beams[i]);
            if ((i+1) % NUM_BEAMS == 0)
                fprintf(signal2, "\n");            
        }
        
        fprintf(signal2, "\ne\n");        
        fflush(signal2);

        fprintf(signal3, "unset key\n");
        fprintf(signal3, "set pm3d\n");
        fprintf(signal3, "set view map\n");
        fprintf(signal3, "set xrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal3, "set yrange [ -0.5 : %f ] \n", NUM_BEAMS - 0.5f);
        fprintf(signal3, "plot '-' matrix with image\n");
        
        for(int i = 2 * NUM_BEAMS * NUM_BEAMS; i < 3 * NUM_BEAMS * NUM_BEAMS; i++)
        {
            fprintf(signal3, "%f ", data->beams[i]);            
            if ((i+1) % NUM_BEAMS == 0)
                fprintf(signal3, "\n");            
        }
        
        fprintf(signal3, "\ne\n");        
        fflush(signal3);*/

        // Display the buffered changes to stdout in the terminal
        //fflush(stdout);
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
    return 0;
}