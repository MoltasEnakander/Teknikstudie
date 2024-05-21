#include "beamformer_wideband.h"

#include <chrono>
#include <ctime>
#include <unistd.h>


void free_resources(beamformingData* data){
    // free allocated memory    
    //free(data->summedSignals);
    free(data->a);
    free(data->b);
    free(data->alpha);
    free(data->beta);    
    free(data->beams);    
    free(data->ordbuffer);    
    fftwf_free(data->fft_data);    
    fftwf_free(data->firfiltersfft);    
    fftwf_free(data->filtered_data);    

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_destroy_plan(data->forw_plans[i]);
        //fftwf_destroy_plan(data->back_plans[i]);
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
    const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
    void* userData
) {
    // Cast our input buffer to a float pointer (since our sample format is `paFloat32`)
    float* in = (float*)inputBuffer;

    // We will not be modifying the output buffer. This line is a no-op.
    (void)outputBuffer;

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
            if (j < FRAMES_PER_BUFFER)
                data->ordbuffer[i * BLOCK_LEN + j] = in[j * NUM_CHANNELS + i];
            else
                data->ordbuffer[i * BLOCK_LEN + j] = 0.0f; // zero-pad 
        }        
    }

    for (int i = 0; i < NUM_CHANNELS; ++i) // calculate fft for each channel
    {
        fftwf_execute(data->forw_plans[i]);
    }

    // perform multiplication in freq domain
    int resultID, filterID, dataID;
    for (int i = 0; i < NUM_CHANNELS; ++i) // for every channel
    {
        for (int j = 0; j < NUM_FILTERS; ++j) // for every filter
        {
            for (int k = 0; k < FFT_OUTPUT_SIZE; ++k) // for all samples
            {                
                // k denotes frequency bin
                // j denotes the filter
                // i denotes the channel
                resultID = k + j * FFT_OUTPUT_SIZE + i * FFT_OUTPUT_SIZE * NUM_FILTERS;
                filterID = k + j * FFT_OUTPUT_SIZE;
                dataID = k + i * FFT_OUTPUT_SIZE;
                data->filtered_data[resultID][0] = data->fft_data[dataID][0] * data->firfiltersfft[filterID][0] - data->fft_data[dataID][1] * data->firfiltersfft[filterID][1];
                data->filtered_data[resultID][1] = data->fft_data[dataID][0] * data->firfiltersfft[filterID][1] - data->fft_data[dataID][1] * data->firfiltersfft[filterID][0];
            }
        }
    }

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    // inverse transform back to time domain    

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
    beamforming<<<numBlocks, threadsPerBlock>>>(data->buffer, data->d_beams, data->a, data->b, data->alpha, data->beta, data->summedSignals);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    cudaMemcpy(data->h_beams, data->d_beams, NUM_VIEWS*NUM_VIEWS*sizeof(float), cudaMemcpyDeviceToHost);

    int maxID = 0;
    float maxVal = data->h_beams[0];

    for (int i = 1; i < NUM_VIEWS * NUM_VIEWS; i++)
    {
        if (maxVal < data->h_beams[i]){
            maxID = i;
            maxVal = data->h_beams[i];
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
    
    py::list res = my_func(NUM_FILTERS, NUM_TAPS, BANDWIDTH); // create the filters
    // temporary save state of data
    std::vector<float> taps;
    for (py::handle obj : res) {  // iterators!
        taps.push_back(obj.attr("__float__")().cast<float>());
    }

    // transfer data for real, goal is to get a buffer that looks like (with zero-padded signals):
    // filter1[0], filter1[1], ..., 0, 0, 0, filter2[0], filter2[1], ..., 0, 0, 0
    // -------- BLOCK_LEN samples ---------, -------- BLOCK_LEN samples --------- 
    float* firfilters = (float*)malloc(BLOCK_LEN * NUM_FILTERS * sizeof(float));
    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        for (int j = 0; j < BLOCK_LEN; ++j)
        {
            if (j < NUM_TAPS)
                firfilters[i * BLOCK_LEN + j] = taps[NUM_TAPS * i + j];
            else
                firfilters[i * BLOCK_LEN + j] = 0.0f; // zero pad filters
        }
    }
    taps.clear();

    // apply fft to filters
    data->firfiltersfft = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_FILTERS * sizeof(fftwf_complex));
    fftwf_plan filter_plans[NUM_FILTERS];
    for (int i = 0; i < NUM_FILTERS; ++i) // create the plans for calculating the fft of each filter block
    {
        filter_plans[i] = fftwf_plan_dft_r2c_1d(BLOCK_LEN, &firfilters[i * BLOCK_LEN], &data->firfiltersfft[i * FFT_OUTPUT_SIZE], FFTW_ESTIMATE);
    }

    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_execute(filter_plans[i]);
    }
    
    for (int i = 0; i < NUM_FILTERS; ++i)
    {
        fftwf_destroy_plan(filter_plans[i]);
    }
    free(firfilters);

    printf("Setting up interpolation data.\n");
    float* theta = linspace(MIN_VIEW, NUM_VIEWS);
    float* phi = linspace(MIN_VIEW, NUM_VIEWS);
    float* delay = calcDelays(theta, phi);
    data->a = calca(delay);
    data->b = calcb(data->a);
    data->alpha = calcalpha(delay, data->b);
    data->beta = calcbeta(data->alpha);
    free(theta); free(phi); free(delay); // free memory which does not have to be allocated anymore
    
    data->beams = (float*)malloc(NUM_VIEWS * NUM_VIEWS * NUM_FILTERS * sizeof(float));
    data->ordbuffer = (float*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(float));
    data->fft_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));
    data->filtered_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * NUM_FILTERS * sizeof(fftwf_complex));

    printf("Creating fft plans.\n");
    for (int i = 0; i < NUM_CHANNELS; ++i) // create the plans for calculating the fft of each channel block
    {
        data->forw_plans[i] = fftwf_plan_dft_r2c_1d(BLOCK_LEN, &data->ordbuffer[i * BLOCK_LEN], &data->fft_data[i * FFT_OUTPUT_SIZE], FFTW_ESTIMATE);
        for (int j = 0; j < NUM_FILTERS; ++j) // for each channel, there are NUM_FILTERS to apply, each application will need FFT_OUPUT_SIZE spots in the array
        {
            //data->back_plans[i] = fftwf_plan_dft_c2r_1d(BLOCK_LEN, &data->filtered_data[i * NUM_FILTERS * FFT_OUTPUT_SIZE + j * FFT_OUTPUT_SIZE], &data->ordbuffer[i * BLOCK_LEN], FFTW_ESTIMATE);
        }
    }
    
    printf("Defining stream paramters.\n");
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
    checkErr(err, data);

    FILE* signal = popen("gnuplot", "w");    

    std::vector<int> bins;
    for (int i = 0; i < FFT_OUTPUT_SIZE; ++i)
    {
        bins.push_back(i);
    }

    std::vector<float> ch1(FFT_OUTPUT_SIZE), lpf(FFT_OUTPUT_SIZE), res1(FFT_OUTPUT_SIZE); 
    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        for (int i = 0; i < FFT_OUTPUT_SIZE; ++i)
        {
            ch1.at(i) = sqrt(data->fft_data[i][0] * data->fft_data[i][0] + data->fft_data[i][1] * data->fft_data[i][1]);
            lpf.at(i) = sqrt(data->firfiltersfft[i][0] * data->firfiltersfft[i][0] + data->firfiltersfft[i][1] * data->firfiltersfft[i][1]);
            res1.at(i) = sqrt(data->filtered_data[i][0] * data->filtered_data[i][0] + data->filtered_data[i][1] * data->filtered_data[i][1]);
        }

        plt::figure(10);
        plt::title("Frequency contents, channel 1");
        plt::clf();    
        plt::plot(bins, ch1);
        plt::xlabel("freq bin");

        plt::figure(11);
        plt::title("Frequency contents, channel 1");
        plt::clf();    
        plt::plot(bins, lpf);
        plt::xlabel("freq bin");

        plt::figure(12);
        plt::title("Frequency contents, channel 1");
        plt::clf();    
        plt::plot(bins, res1);
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
            fprintf(signal, "%f ", data->h_beams[i]);
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

float* calcDelays(float* theta, float* phi)
{
    float* d = (float*)malloc(NUM_VIEWS*NUM_VIEWS*NUM_CHANNELS*sizeof(float));
    printf("Ha1\n");

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