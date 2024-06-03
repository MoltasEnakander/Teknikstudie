#include "beamformer2.h"

#include <chrono>
#include <ctime>
#include <unistd.h>

#include <thread>


void free_resources(beamformingData* data)
{
    // free allocated memory
    free(data->beams);
    fftwf_free(data->ordbuffer);
    fftwf_free(data->block);
    free(data->summedSignals);    
    free(data->a);
    free(data->alpha);
    free(data->b);
    free(data->beta);
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
    int resultID, filterID, dataID;
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

    // create beams
    int i, j, k;
    double beamStrength;    

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

    // recursive bandpass filters

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
    lp_filter_plan = fftwf_plan_dft_1d(FFTW_ESTIMATE, lpfilter, data->LP_filter, FFTW_FORWARD, FFTW_ESTIMATE);

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

    printf("Setting up buffers.\n");
    float* theta = linspace(MIN_VIEW, NUM_VIEWS);
    float* phi = linspace(MIN_VIEW, NUM_VIEWS);
    float* delay = calcDelays(theta, phi);

    data->a = calca(delay);
    data->b = calcb(data->a);
    data->alpha = calcalpha(delay, data->b);
    data->beta = calcbeta(data->alpha);    
    
    free(theta); free(phi); free(delay); // free memory which does not have to be allocated anymore*/    
    
    data->beams = (float*)malloc(NUM_VIEWS * NUM_VIEWS * NUM_FILTERS * sizeof(float));
    data->ordbuffer = (fftwf_complex*)fftwf_malloc(FRAMES_PER_BUFFER * NUM_CHANNELS * sizeof(fftwf_complex));
    data->block = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));
    data->summedSignals = (float*)malloc(BLOCK_LEN * NUM_VIEWS * NUM_VIEWS * sizeof(float));

    std::memset(data->summedSignals, 0, BLOCK_LEN * NUM_VIEWS * NUM_VIEWS * sizeof(float));

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
    for (int i = 0; i < 4; ++i)
    {
        callBack(&input[i * FRAMES_PER_BUFFER], data);
    }

    free(input);
    
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