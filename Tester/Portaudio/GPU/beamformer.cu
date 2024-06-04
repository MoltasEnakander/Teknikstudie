#include "beamformer.h"

#include <chrono>
#include <ctime>
#include <unistd.h>

#include <thread>

/*for (i = 0; i < NUM_VIEWS * NUM_VIEWS; ++i) // loop beams
{        
    beamStrength = 0;
    for (j = 0; j < NUM_CHANNELS; ++j) // loop channels
    {
        // interpolation of left sample
        for (k = std::max(-data->a[i * NUM_CHANNELS + j], 0); k < std::min(BLOCK_LEN - data->a[i * NUM_CHANNELS + j], BLOCK_LEN); k++)
        {
            data->summedSignals[i * BLOCK_LEN + k] += data->alpha[i * NUM_CHANNELS + j] * data->filtered_data_temp[k + data->a[i * NUM_CHANNELS + j] + j * BLOCK_LEN][0];
        }

        // interpolation of right sample
        for (k = std::max(-data->b[i * NUM_CHANNELS + j], 0); k < std::min(BLOCK_LEN - data->b[i * NUM_CHANNELS + j], BLOCK_LEN); k++)
        {   
            data->summedSignals[i * BLOCK_LEN + k] += data->beta[i * NUM_CHANNELS + j] * data->filtered_data_temp[k + data->b[i * NUM_CHANNELS + j] + j * BLOCK_LEN][0];
        }
    }
    
    // normalize and calculate "strength" of beam
    for (k = 0; k < BLOCK_LEN; k++)
    {
        data->summedSignals[i * BLOCK_LEN + k] /= NUM_CHANNELS;
        data->summedSignals[i * BLOCK_LEN + k] = data->summedSignals[i * BLOCK_LEN + k] * data->summedSignals[i * BLOCK_LEN + k] / BLOCK_LEN;
        beamStrength += data->summedSignals[i * BLOCK_LEN + k]; 
    }

    data->beams[i] = 10 * std::log10(beamStrength);        
}*/

__global__
void interpolateChannels(const cufftComplex* inputBuffer, cufftComplex* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta)
{
    int id;    
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index of this thread
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN; // global index of this thread

    // l1 -> 0 - 2047
    // l2 -> 0 - 2047 + i * 2048, i -> 0 - 168

    for (int k = 0; k < NUM_CHANNELS; ++k)
    {
        id = k + i * NUM_CHANNELS;        
        if (max(0, -a[id]) == 0 && l1 < BLOCK_LEN - a[id]) // a >= 0            
            summedSignals[l2].x += alpha[id] * inputBuffer[l1 + a[id] + k * BLOCK_LEN].x; // do not write to the a[id] end positions
        else if (max(0, -a[id]) > 0 && l1 >= a[id]) 
            summedSignals[l2].x += alpha[id] * inputBuffer[l1 + a[id] + k * BLOCK_LEN].x; // do not write to the first a[id]-1 positions

        if (max(0, -b[id]) == 0 && l1 < BLOCK_LEN - b[id]) // b >= 0
            summedSignals[l2].x += beta[id] * inputBuffer[l1 + b[id] + k * BLOCK_LEN].x; // do not write to the b[id] end positions
        else if (max(0, -b[id]) > 0 && l1 >= b[id]) 
            summedSignals[l2].x += beta[id] * inputBuffer[l1 + b[id] + k * BLOCK_LEN].x; // do not write to the first b[id]-1 positions*/
    }    
}

__global__ 
void beamforming(const cufftComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftComplex* summedSignals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_VIEWS * NUM_VIEWS){
        return;
    }

    // interpolate channels    
    interpolateChannels<<<(BLOCK_LEN+255)/256, 256>>>(inputBuffer, summedSignals, i, a, b, alpha, beta);
    cudaDeviceSynchronize();    
}

__global__
void bandpass_filtering_calcs(int i, cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter)
{
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index, from 0 to BLOCK_LEN - 1
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN; // internal index + compensation for which beam is being calced
    int l3 = blockIdx.x * blockDim.x + threadIdx.x + i * NUM_FILTERS * BLOCK_LEN; // as l2, but compensates for beams being calced in different freq-bands
    //       -           0 - 2047               -, + i *     6      *  2048   

    for (int j = 0; j < NUM_FILTERS; ++j)
    {        
        summedSignals_fft_BP[l3 + j * BLOCK_LEN].x = summedSignals_fft[l2].x * BP_filter[l1 + j * BLOCK_LEN].x - summedSignals_fft[l2].y * BP_filter[l1 + j * BLOCK_LEN].y;
        summedSignals_fft_BP[l3 + j * BLOCK_LEN].y = summedSignals_fft[l2].x * BP_filter[l1 + j * BLOCK_LEN].y + summedSignals_fft[l2].y * BP_filter[l1 + j * BLOCK_LEN].x;        
    }
    // after these calculations there should be NUM_FILTERS signals per view, and each signals is BLOCK_LEN samples long, the strength of the signals need to be calced
}

__global__
void bandpass_filtering(cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter, float* beams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;    

    if (i >= NUM_VIEWS * NUM_VIEWS){
        return;
    }

    // calculations
    bandpass_filtering_calcs<<<(BLOCK_LEN+255)/256, 256>>>(i, summedSignals_fft_BP, summedSignals_fft, BP_filter);
    cudaDeviceSynchronize();

    if (i == 0){
        beams[0] = 4.0f;
    }

    /*float beamstrength;
    int id;
    for (int j = 0; j < NUM_FILTERS; ++j)
    {
        beamstrength = 0.0f;

        for (int k = 0; k < BLOCK_LEN; ++k)
        {
            id = k + j * NUM_FILTERS + i * NUM_FILTERS * BLOCK_LEN;
            beamstrength += sqrt(summedSignals_fft_BP[id].x * summedSignals_fft_BP[id].x + summedSignals_fft_BP[id].y * summedSignals_fft_BP[id].y);
        }
        beams[i + j * NUM_VIEWS * NUM_VIEWS] = beamstrength;//10 * log10(beamstrength);
    }*/
}


void free_resources(beamformingData* data)
{
    // free allocated memory
    free(data->beams);
    fftwf_free(data->ordbuffer);
    fftwf_free(data->block);
    cudaFree(data->summedSignals);   
    cudaFree(data->summedSignals_fft);
    cudaFree(data->summedSignals_fft_BP);
    cudaFree(data->BP_filter);
    cudaFree(data->a);
    cudaFree(data->alpha);
    cudaFree(data->b);
    cudaFree(data->beta);
    cudaFree(data->gpu_block);
    cudaFree(data->gpu_beams);
    fftwf_free(data->fft_data);
    //fftwf_free(data->firfiltersfft);
    fftwf_free(data->filtered_data);
    fftwf_free(data->filtered_data_temp);
    fftwf_free(data->LP_filter);        
    
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
/*static void checkErr(PaError err, beamformingData* data) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        free_resources(data);
        exit(EXIT_FAILURE);
    }
}*/

// PortAudio stream callback function. Will be called after every
// FRAMES_PER_BUFFER audio samples PortAudio captures. Used to process the
// resulting audio sample.
/*static int streamCallback(
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
)*/
static void callBack(fftwf_complex* inputBuffer, beamformingData* data)
{    
    // Cast our input buffer to a float pointer (since our sample format is `paFloat32`)
    fftwf_complex* in = (fftwf_complex*)inputBuffer;

    // We will not be modifying the output buffer. This line is a no-op.
    //(void)outputBuffer;

    /*beamformingData* data = (beamformingData*)userData;
    
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
    }*/

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    for (int i = 0; i < NUM_CHANNELS; ++i) // sort the incoming buffer based on channel
    {       
        for (int j = 0; j < FRAMES_PER_BUFFER; ++j)
        {            
            //data->ordbuffer[i * BLOCK_LEN + j] = in[j * NUM_CHANNELS + i];
            data->ordbuffer[i * FRAMES_PER_BUFFER + j][0] = in[j][0];
        }        
    }

    for (int i = 0; i < NUM_CHANNELS; ++i) // save some of the old data and add the new buffer
    {
        std::memcpy(&(data->block[i * BLOCK_LEN]), &(data->block[i * BLOCK_LEN + FRAMES_PER_BUFFER]), (BLOCK_LEN - FRAMES_PER_BUFFER) * sizeof(fftwf_complex));
        std::memcpy(&(data->block[i * BLOCK_LEN + (BLOCK_LEN - FRAMES_PER_BUFFER)]), &(data->ordbuffer[i * FRAMES_PER_BUFFER]), FRAMES_PER_BUFFER * sizeof(fftwf_complex));
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
        fftwf_execute(data->back_plans[i]);    
    }

    cudaMemcpy(data->gpu_block, data->filtered_data_temp, BLOCK_LEN*NUM_CHANNELS*sizeof(fftwf_complex), cudaMemcpyHostToDevice); // copy buffer to GPU memory

    // beamform
    int numBlocks;
    dim3 threadsPerBlock;
    if (NUM_VIEWS * NUM_VIEWS > MAX_THREADS_PER_BLOCK){
        numBlocks = (NUM_VIEWS * NUM_VIEWS) % MAX_THREADS_PER_BLOCK + 1;
        threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK);
    }
    else{
        numBlocks = 1;
        threadsPerBlock = dim3(NUM_VIEWS * NUM_VIEWS);
    }
    
    cudaMemset(data->summedSignals, 0, BLOCK_LEN * NUM_VIEWS * NUM_VIEWS * sizeof(cufftComplex)); // reset signals    

    // create beams    
    beamforming<<<numBlocks, threadsPerBlock>>>(data->gpu_block, data->a, data->b, data->alpha, data->beta, data->summedSignals);
    cudaDeviceSynchronize();

    cufftExecC2C(data->planMany, data->summedSignals, data->summedSignals_fft, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    
    bandpass_filtering<<<numBlocks, threadsPerBlock>>>(data->summedSignals_fft_BP, data->summedSignals_fft, data->BP_filter, data->gpu_beams);
    cudaDeviceSynchronize();    

    // copy the intensity of the beams to the cpu
    cudaMemcpy(data->beams, data->gpu_beams, NUM_VIEWS*NUM_VIEWS*NUM_FILTERS*sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f \n", data->beams[0]);    

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    //return finished;
}

void listen_live() 
{
    // Initialize PortAudio
    /*PaError err;
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

    printf("Device = %d\n", device);*/
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------    

    // setup necessary data containers for the beamforming
    beamformingData* data = (beamformingData*)malloc(sizeof(beamformingData));
    data->maxFrameIndex = NUM_SECONDS * SAMPLE_RATE; // Record for a few seconds.
    data->frameIndex = 0;

    printf("Setting up fir filters.\n");    
    py::scoped_interpreter python;

    py::function my_func =
        py::reinterpret_borrow<py::function>(
            py::module::import("filtercreation").attr("filtercreation")  
    );    
    
    /*py::list res = my_func(NUM_FILTERS, NUM_TAPS, BANDWIDTH); // create the filters
    // temporary save state of data
    std::vector<float> taps;
    for (py::handle obj : res) {  // iterators!
        taps.push_back(obj.attr("__float__")().cast<float>());
    }*/

    py::list res2 = my_func(1, NUM_TAPS, 10000.0f / 22050.0f);
    // temporary save state of data
    std::vector<float> taps2;
    for (py::handle obj : res2) {  // iterators!
        taps2.push_back(obj.attr("__float__")().cast<float>());
    }

    // transfer data for real, goal is to get a buffer that looks like (with zero-padded signals):
    // filter1[0], filter1[1], ..., 0, 0, 0, filter2[0], filter2[1], ..., 0, 0, 0
    // -------- BLOCK_LEN samples ---------, -------- BLOCK_LEN samples --------- 
    /*fftwf_complex* firfilters = (fftwf_complex*)malloc(BLOCK_LEN * NUM_FILTERS * sizeof(fftwf_complex));
    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        for (int j = 0; j < BLOCK_LEN; ++j)
        {
            if (j < NUM_TAPS)
                firfilters[i * BLOCK_LEN + j][0] = taps[NUM_TAPS * i + j];
            else
                firfilters[i * BLOCK_LEN + j][0] = 0.0f; // zero pad filters
            firfilters[i * BLOCK_LEN + j][1] = 0.0f;
        }
    }
    taps.clear();*/

    fftwf_complex* lpfilter = (fftwf_complex*)malloc(BLOCK_LEN * sizeof(fftwf_complex));
    for (int i = 0; i < BLOCK_LEN; ++i)
    {
        if (i < NUM_TAPS)
            lpfilter[i][0] = taps2[i];            
        else
            lpfilter[i][0] = 0.0f; // zero pad filters            
        lpfilter[i][1] = 0.0f;
    }
    taps2.clear();    

    // apply fft to filters
    //data->firfiltersfft = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_FILTERS * sizeof(fftwf_complex));
    data->LP_filter = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * sizeof(fftwf_complex));
    //fftwf_plan filter_plans[NUM_FILTERS];
    fftwf_plan lp_filter_plan;
    /*for (int i = 0; i < NUM_FILTERS; ++i) // create the plans for calculating the fft of each filter block
    {
        filter_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &firfilters[i * BLOCK_LEN], &data->firfiltersfft[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE);
    }*/
    lp_filter_plan = fftwf_plan_dft_1d(BLOCK_LEN, lpfilter, data->LP_filter, FFTW_FORWARD, FFTW_ESTIMATE);

    /*for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_execute(filter_plans[i]);
    }*/
    fftwf_execute(lp_filter_plan);
    
    /*for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_destroy_plan(filter_plans[i]);
    }*/
    fftwf_destroy_plan(lp_filter_plan);

    //free(firfilters);
    free(lpfilter);
    printf("FIR filters are created.\n");

    float* theta = linspace(MIN_VIEW, NUM_VIEWS);
    float* phi = linspace(MIN_VIEW, NUM_VIEWS);
    float* delay = calcDelays(theta, phi);

    int* a = calca(delay);
    int* b = calcb(a);
    float* alpha = calcalpha(delay, b);
    float* beta = calcbeta(alpha);

    cudaMalloc(&(data->a), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->b), sizeof(int) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->alpha), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMalloc(&(data->beta), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_CHANNELS);
    cudaMemcpy(data->a, a, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->b, b, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data->alpha, alpha, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->beta, beta, NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
    free(theta); free(phi); free(delay); free(a); free(b); free(alpha); free(beta); // free memory which does not have to be allocated anymore*/    

    data->beams = (float*)malloc(NUM_VIEWS * NUM_VIEWS * NUM_FILTERS * sizeof(float));
    std::memset(data->beams, 0.0f, NUM_VIEWS * NUM_VIEWS * NUM_FILTERS * sizeof(float));
    cudaMalloc(&(data->gpu_beams), sizeof(float) * NUM_VIEWS * NUM_VIEWS * NUM_FILTERS);

    cudaMalloc(&(data->gpu_block), sizeof(cufftComplex) * NUM_CHANNELS * BLOCK_LEN);

    data->ordbuffer = (fftwf_complex*)fftwf_malloc(FRAMES_PER_BUFFER * NUM_CHANNELS * sizeof(fftwf_complex));
    data->block = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));
    data->summedSignals2 = (fftwf_complex*)fftwf_malloc(NUM_VIEWS * NUM_VIEWS * BLOCK_LEN * sizeof(fftwf_complex));
    cudaMalloc(&(data->summedSignals), sizeof(cufftComplex) * NUM_VIEWS * NUM_VIEWS * BLOCK_LEN);
    cudaMalloc(&(data->summedSignals_fft), sizeof(cufftComplex) * NUM_VIEWS * NUM_VIEWS * BLOCK_LEN);
    cudaMalloc(&(data->summedSignals_fft_BP), sizeof(cufftComplex) * NUM_VIEWS * NUM_VIEWS * BLOCK_LEN * NUM_FILTERS);

    cudaMemset(data->summedSignals_fft_BP, 1.0f, sizeof(cufftComplex) * NUM_VIEWS * NUM_VIEWS * BLOCK_LEN * NUM_FILTERS);
    cudaMalloc(&(data->BP_filter), sizeof(cufftComplex) * BLOCK_LEN * NUM_FILTERS);

    cudaMemset(data->summedSignals, 0, BLOCK_LEN * NUM_VIEWS * NUM_VIEWS * sizeof(cufftComplex));
    cudaMemset(data->BP_filter, 0, BLOCK_LEN * NUM_FILTERS * sizeof(cufftComplex));

    data->fft_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));
    data->filtered_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));
    data->filtered_data_temp = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));

    for (int i = 0; i < FRAMES_PER_BUFFER * NUM_CHANNELS; ++i)
    {
        data->ordbuffer[i][0] = 0.0f;
        data->ordbuffer[i][1] = 0.0f;
    }
    
    for (int i = 0; i < BLOCK_LEN * NUM_CHANNELS; ++i)
    {
        data->block[i][0] = 0.0f;
        data->block[i][1] = 0.0f;
    }

    printf("Creating fft plans.\n");
    int n[1] = {BLOCK_LEN};
    int inembed[] = {BLOCK_LEN};
    int onembed[] = {BLOCK_LEN};
    
    cufftPlanMany(&(data->planMany), 1, n, inembed, 1, BLOCK_LEN, onembed, 1, BLOCK_LEN, CUFFT_C2C, NUM_VIEWS*NUM_VIEWS);

    for (int i = 0; i < NUM_CHANNELS; ++i) // create the plans for calculating the fft of each channel block
    {
        data->forw_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &data->block[i * BLOCK_LEN], &data->fft_data[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE); // NUM_CHANNELS channels for each block which requires FFT_OUTPUT_SIZE spots to store the fft data
        data->back_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &data->filtered_data[i * FFT_OUTPUT_SIZE], &data->filtered_data_temp[i * BLOCK_LEN], FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    // Create test signal, each channel will get the exact same signal for now
    fftwf_complex* input = (fftwf_complex*)malloc(FRAMES_PER_BUFFER * 4 * sizeof(fftwf_complex));
    for (int i = 0; i < FRAMES_PER_BUFFER * 4; ++i)
    {
        input[i][0] = cosf(2 * M_PI * 520.0f * (1.0f / SAMPLE_RATE) * i);// + cosf(2 * M_PI * 1700.0f * (1.0f / SAMPLE_RATE) * i) + \
                    cosf(2 * M_PI * 2750.0f * (1.0f / SAMPLE_RATE) * i) + cosf(2 * M_PI * 3400.0f * (1.0f / SAMPLE_RATE) * i);
        input[i][1] = 0.0f;
    }    

    // run the callback function 8 times
    for (int i = 0; i < 36; ++i)
    {
        callBack(&input[0], data);
    }

    free(input);

    std::vector<float> d(BLOCK_LEN), bins(BLOCK_LEN), in(BLOCK_LEN), LP(BLOCK_LEN), filt(BLOCK_LEN);

    for (int i = 0; i < BLOCK_LEN; ++i)
    {
        bins.at(i) = i;
        in.at(i) = sqrt(data->fft_data[i][0] * data->fft_data[i][0] + data->fft_data[i][1] * data->fft_data[i][1]);
        LP.at(i) = sqrt(data->LP_filter[i][0] * data->LP_filter[i][0] + data->LP_filter[i][1] * data->LP_filter[i][1]);
        filt.at(i) = sqrt(data->filtered_data[i][0] * data->filtered_data[i][0] + data->filtered_data[i][1] * data->filtered_data[i][1]);
    }

    /*plt::figure(10);
    plt::title("Frequency contents, channel 1");
    plt::clf();    
    plt::plot(bins, in);
    plt::xlabel("freq bin");

    plt::figure(11);
    plt::title("Frequency contents, channel 1");
    plt::clf();    
    plt::plot(bins, LP);
    plt::xlabel("freq bin");

    plt::figure(12);
    plt::title("Frequency contents, channel 1");
    plt::clf();    
    plt::plot(bins, filt);
    plt::xlabel("freq bin");

    plt::show();*/
    
    /*printf("Defining stream parameters.\n");
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
        FRAMES_PER_BUFFER,
        paNoFlag,
        streamCallback,
        data
    );
    checkErr(err, data);

    // Begin capturing audio
    err = Pa_StartStream(stream);
    checkErr(err, data);*/

    //FILE* signal = popen("gnuplot", "w");    
    
    //while( ( err = Pa_IsStreamActive( stream ) ) == 1 )    
    //{ 
        //plt::pause(2.0);

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
        plt::pause(0.02);*/
        


        // plot beamforming results in color map
        /*fprintf(signal, "unset key\n");
        fprintf(signal, "set pm3d\n");
        fprintf(signal, "set view map\n");
        fprintf(signal, "set xrange [ -0.5 : %f ] \n", NUM_VIEWS-0.5);
        fprintf(signal, "set yrange [ -0.5 : %f ] \n", NUM_VIEWS-0.5);
        fprintf(signal, "plot '-' matrix with image\n");
        int c = 0;
        for(int i = 0; i < NUM_VIEWS * NUM_VIEWS; i += NUM_FILTERS) // plot map for the lowest frequency band    
        {
            fprintf(signal, "%f ", data->beams[i]);
            c++;
            if (c % NUM_VIEWS == 0)
                fprintf(signal, "\n");            
        }
        
        fprintf(signal, "e\n");
        fprintf(signal, "e\n");
        fflush(signal);

        // Display the buffered changes to stdout in the terminal
        //fflush(stdout);

        //plt::show();
    //}    

    // Stop capturing audio
    /*err = Pa_StopStream(stream);
    checkErr(err, data);

    // Close the PortAudio stream
    err = Pa_CloseStream(stream);
    checkErr(err, data);

    // Terminate PortAudio
    err = Pa_Terminate();
    checkErr(err, data);*/

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
             
    // now assign the values to the array
    for (int i = 0; i < num; i++)
    {
        f[i] = (a + i * VIEW_INTERVAL) * M_PI / 180.0f;
    }
    return f;
}

float* calcDelays(float* theta, float* phi)
{
    float* d = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));    

    int pid = 0; // phi index
    int tid = 0; // theta index
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

int* calca(float* delay)
{
    int* a = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        a[i] = floor(delay[i]);
    }
    return a;
}

int* calcb(int* a)
{
    int* b = (int*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        b[i] = a[i] + 1;
    }
    return b;
}

float* calcalpha(float* delay, int* b)
{
    float* alpha = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        alpha[i] = b[i] - delay[i];
    }
    return alpha;
}

float* calcbeta(float* alpha)
{
    float* beta = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS; ++i)
    {
        beta[i] = 1 - alpha[i];
    }
    return beta;
}