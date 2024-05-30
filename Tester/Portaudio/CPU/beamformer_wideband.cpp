#include "beamformer_wideband.h"

#include <chrono>
#include <ctime>
#include <unistd.h>

#include <thread>


void free_resources(beamformingData* data){
    // free allocated memory    
    //free(data->summedSignals);
    free(data->beams);    
    free(data->ordbuffer);    
    //free(data->cosines);
    free(data->cosine_counter);
    fftwf_free(data->fft_data);
    fftwf_free(data->firfiltersfft);
    fftwf_free(data->filtered_data);
    fftwf_free(data->filtered_data_temp);
    fftwf_free(data->OLA_signal);
    fftwf_free(data->OLA_fft);
    fftwf_free(data->LP_filter);    
    free(data->OLA_decimated);

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_destroy_plan(data->forw_plans[i]);
        for (int j = 0; j < NUM_FILTERS; ++j)
        {
            fftwf_destroy_plan(data->back_plans[i * NUM_FILTERS + j]);
            fftwf_destroy_plan(data->OLA_forw[i * NUM_FILTERS + j]);
            fftwf_destroy_plan(data->OLA_back[i * NUM_FILTERS + j]);
        }
        
    }

    free(data);
}

void shift(beamformingData *data, int i){
    // OLA_signal is ordered by channel first, then filter, then blocks
    // shift out previous block and add the newest block at the end
    float f_c;

    for (int j = 0; j < NUM_FILTERS; ++j)
    {
        f_c = F_C + 2 * F_C * j;
        // shift everything FRAMES_PER_BUFFER to the left            
        std::memcpy(&data->OLA_signal[j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH], \
                    &data->OLA_signal[FRAMES_PER_BUFFER + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH], \
                    (NUM_OLA_BLOCK - 1) * FRAMES_PER_BUFFER * sizeof(fftwf_complex));

        int cosine_id = data->cosine_block * FRAMES_PER_BUFFER;
        // shift in last part and add new part
        for (int l = 0; l < BLOCK_LEN - FRAMES_PER_BUFFER; ++l) // the beginning of the new, will be overlapped with the end of the previous block
        {
            data->OLA_signal[l + (NUM_OLA_BLOCK - 1) * FRAMES_PER_BUFFER + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][0] = \
            data->OLA_signal[l + NUM_OLA_BLOCK * FRAMES_PER_BUFFER + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][0] + \
            data->filtered_data_temp[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN][0] / BLOCK_LEN * \
            0.5f * cosf(2.0f * M_PI * f_c * (1.0f/SAMPLE_RATE) * data->cosine_counter[j + i * NUM_FILTERS]);
            // new block needs to be mult. with a cosine to achieve IQ down-conversion, factor 0.5 since it will be overlapped with another part that also has factor 0.5
            data->cosine_counter[j + i * NUM_FILTERS]++;
            if (data->cosine_counter[j + i * NUM_FILTERS] >= (int)SAMPLE_RATE)
                data->cosine_counter[j + i * NUM_FILTERS] = 0;
            
        }
        for (int l = BLOCK_LEN - FRAMES_PER_BUFFER; l < FRAMES_PER_BUFFER; ++l) // the middle of the new block, no overlap
        {
            data->OLA_signal[l + (NUM_OLA_BLOCK - 1) * FRAMES_PER_BUFFER + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][0] = \
            data->filtered_data_temp[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN][0] / BLOCK_LEN * \
            1.0f * cosf(2.0f * M_PI * f_c * (1.0f/SAMPLE_RATE) * data->cosine_counter[j + i * NUM_FILTERS]);
            // new block needs to be mult. with a cosine to achieve IQ down-conversion, factor 1.0 since no overlap
            data->cosine_counter[j + i * NUM_FILTERS]++;
            if (data->cosine_counter[j + i * NUM_FILTERS] >= (int)SAMPLE_RATE)
                data->cosine_counter[j + i * NUM_FILTERS] = 0;
        }
        for (int l = FRAMES_PER_BUFFER; l < BLOCK_LEN; ++l) // the end of the new block, will be overlapped with the beginning of the next block
        {
            data->OLA_signal[l + (NUM_OLA_BLOCK - 1) * FRAMES_PER_BUFFER + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][0] = \
            data->filtered_data_temp[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN][0] / BLOCK_LEN * \
            0.5f * cosf(2.0f * M_PI * f_c * (1.0f/SAMPLE_RATE) * data->cosine_counter[j + i * NUM_FILTERS]);
            // new block needs to be mult. with a cosine to achieve IQ down-conversion, factor 0.5 for previously stated reason
            data->cosine_counter[j + i * NUM_FILTERS]++;
            if (data->cosine_counter[j + i * NUM_FILTERS] >= (int)SAMPLE_RATE)
                data->cosine_counter[j + i * NUM_FILTERS] = 0;
        }
        data->cosine_counter[j + i * NUM_FILTERS] -= (BLOCK_LEN - FRAMES_PER_BUFFER);
        if (data->cosine_counter[j + i * NUM_FILTERS] < 0)
        {
            data->cosine_counter[j + i * NUM_FILTERS] += (int)SAMPLE_RATE; 
        }
        fftwf_execute(data->OLA_forw[i * NUM_FILTERS + j]);
    }
}

void lpfilter(beamformingData *data, int i){
    float temp1, temp2, temp3, temp4;
    for (int j = 0; j < NUM_FILTERS; ++j)
    {
        for (int k = 0; k < OLA_FFT_OUTPUT_SIZE; ++k)
        {
            temp1 = data->OLA_fft[k + j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE][0];
            temp2 = data->OLA_fft[k + j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE][1];
            temp3 = data->LP_filter[k][0];
            temp4 = data->LP_filter[k][1];

            data->OLA_fft[k + j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE][0] = temp1 * temp3 - temp2 * temp4;
            data->OLA_fft[k + j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE][1] = temp1 * temp4 + temp2 * temp3;            
        }
        //fftwf_execute(data->OLA_back[i * NUM_FILTERS + j]);
    }
}

void decimate(beamformingData *data, int i)
{
    for (int j = 0; j < NUM_FILTERS; ++j)
    {            
        for (int k = 0; k < DECIMATED_LENGTH; ++k)
        {                
            data->OLA_decimated[k + j * DECIMATED_LENGTH + i * NUM_FILTERS * DECIMATED_LENGTH][0] = \
            data->OLA_signal[k * DECIMATED_STEP + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][0];

            data->OLA_decimated[k + j * DECIMATED_LENGTH + i * NUM_FILTERS * DECIMATED_LENGTH][1] = \
            data->OLA_signal[k * DECIMATED_STEP + j * PADDED_OLA_LENGTH + i * NUM_FILTERS * PADDED_OLA_LENGTH][1];
        }
    }
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
        for (int j = 0; j < BLOCK_LEN; ++j)
        {
            if (j < FRAMES_PER_BUFFER)
                //data->ordbuffer[i * BLOCK_LEN + j] = in[j * NUM_CHANNELS + i];
                data->ordbuffer[i * BLOCK_LEN + j][0] = in[j][0];
            else
                data->ordbuffer[i * BLOCK_LEN + j][0] = 0.0f; // zero-pad 
            data->ordbuffer[i * BLOCK_LEN + j][1] = 0.0f; // zero-pad 
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
                data->filtered_data[resultID][1] = data->fft_data[dataID][0] * data->firfiltersfft[filterID][1] + data->fft_data[dataID][1] * data->firfiltersfft[filterID][0];                
            }
            // inverse fourier transform to get back signals in time domain. each block has been filtered through NUM_FILTERS different filters and this is done for each channel
            // there should be (NUM_FILTERS * NUM_CHANNELS) blocks in total
            fftwf_execute(data->back_plans[i * NUM_FILTERS + j]);
        }
    }
    
    // add block to the OLA_signal, divide the work on several threads    
    std::thread th1(shift, data, 0);
    std::thread th2(shift, data, 1);
    std::thread th3(shift, data, 2);
    std::thread th4(shift, data, 3);
    std::thread th5(shift, data, 4);
    std::thread th6(shift, data, 5);
    std::thread th7(shift, data, 6);
    std::thread th8(shift, data, 7);
    std::thread th9(shift, data, 8);
    std::thread th10(shift, data, 9);
    std::thread th11(shift, data, 10);
    std::thread th12(shift, data, 11);
    std::thread th13(shift, data, 12);
    std::thread th14(shift, data, 13);
    std::thread th15(shift, data, 14);
    std::thread th16(shift, data, 15);

    // wait for threads to finish
    th1.join();
    th2.join();
    th3.join();
    th4.join();
    th5.join();
    th6.join();
    th7.join();
    th8.join();
    th9.join();
    th10.join();
    th11.join();
    th12.join();
    th13.join();
    th14.join();
    th15.join();
    th16.join();

    data->cosine_block++;
    data->cosine_block = data->cosine_block % NUM_OLA_BLOCK;

    // apply lp-filter
    /*std::thread th17(lpfilter, data, 0);
    std::thread th18(lpfilter, data, 1);
    std::thread th19(lpfilter, data, 2);
    std::thread th20(lpfilter, data, 3);
    std::thread th21(lpfilter, data, 4);
    std::thread th22(lpfilter, data, 5);
    std::thread th23(lpfilter, data, 6);
    std::thread th24(lpfilter, data, 7);
    std::thread th25(lpfilter, data, 8);
    std::thread th26(lpfilter, data, 9);
    std::thread th27(lpfilter, data, 10);
    std::thread th28(lpfilter, data, 11);
    std::thread th29(lpfilter, data, 12);
    std::thread th30(lpfilter, data, 13);
    std::thread th31(lpfilter, data, 14);
    std::thread th32(lpfilter, data, 15);

    th17.join();
    th18.join();
    th19.join();
    th20.join();
    th21.join();
    th22.join();
    th23.join();
    th24.join();
    th25.join();
    th26.join();
    th27.join();
    th28.join();
    th29.join();
    th30.join();
    th31.join();
    th32.join();*/

    // decimate signals
    // signals are centered around 0Hz with bandwidth BANDWIDTH, new sample-rate needs to be ~ 2 * BANDWIDTH
    // downsample by a factor SAMPLE_RATE / (4 * F_C)    

    /*std::thread th33(decimate, data, 0);
    std::thread th34(decimate, data, 1);
    std::thread th35(decimate, data, 2);
    std::thread th36(decimate, data, 3);
    std::thread th37(decimate, data, 4);
    std::thread th38(decimate, data, 5);
    std::thread th39(decimate, data, 6);
    std::thread th40(decimate, data, 7);
    std::thread th41(decimate, data, 8);
    std::thread th42(decimate, data, 9);
    std::thread th43(decimate, data, 10);
    std::thread th44(decimate, data, 11);
    std::thread th45(decimate, data, 12);
    std::thread th46(decimate, data, 13);
    std::thread th47(decimate, data, 14);
    std::thread th48(decimate, data, 15);

    // wait for threads to finish
    th33.join();
    th34.join();
    th35.join();
    th36.join();
    th37.join();
    th38.join();
    th39.join();
    th40.join();
    th41.join();
    th42.join();
    th43.join();
    th44.join();
    th45.join();
    th46.join();
    th47.join();
    th48.join();*/

    /*float beamstrength; // TODO: beräkna beamstrength
    for (int i = 0; i < NUM_VIEWS * NUM_VIEWS; ++i)
    {
        for (int j = 0; j < NUM_FILTERS; ++j) // this ordering is different from all previous loops which is stupid, but it allows me to sum the channels directly
        {
            beamstrength = 0.0f;
            for (int k = 0; k < NUM_CHANNELS; ++k)
            {
                for (int l = 0; l < DECIMATED_LENGTH; ++l)
                {
                    if (k == 0){ // reset signals
                        data->summedSignals[l][0] = data->OLA_decimated[l + k * DECIMATED_LENGTH * NUM_FILTERS + j * DECIMATED_LENGTH][0] * \
                                                    data->phase_shifts[l + j * DECIMATED_LENGTH + k * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][0] / NUM_CHANNELS;
                        data->summedSignals[l][1] = data->OLA_decimated[l + k * DECIMATED_LENGTH * NUM_FILTERS + j * DECIMATED_LENGTH][1] * \
                                                    data->phase_shifts[l + j * DECIMATED_LENGTH + k * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][1] / NUM_CHANNELS;
                    }
                    else{
                        data->summedSignals[l][0] += data->OLA_decimated[l + k * DECIMATED_LENGTH * NUM_FILTERS + j * DECIMATED_LENGTH][0] * \
                                                    data->phase_shifts[l + j * DECIMATED_LENGTH + k * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][0] / NUM_CHANNELS;
                        data->summedSignals[l][1] += data->OLA_decimated[l + k * DECIMATED_LENGTH * NUM_FILTERS + j * DECIMATED_LENGTH][1] * \
                                                    data->phase_shifts[l + j * DECIMATED_LENGTH + k * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][1] / NUM_CHANNELS;
                    }
                    if (k == NUM_CHANNELS - 1){
                        beamstrength += data->summedSignals[l];
                    }
                }
            }

            data->beams[j + i * NUM_FILTERS] = 10 * std::log10(beamstrength); // each view will have NUM_FILTERS beams listening on different frequency bands      
        }
    }*/

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

    

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
    
    py::list res = my_func(NUM_FILTERS, NUM_TAPS, BANDWIDTH); // create the filters
    // temporary save state of data
    std::vector<float> taps;
    for (py::handle obj : res) {  // iterators!
        taps.push_back(obj.attr("__float__")().cast<float>());
    }

    py::list res2 = my_func(1, NUM_TAPS, BANDWIDTH); // create the lp filter for use after IQ down-conversion
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
                firfilters[i * BLOCK_LEN + j][0] = 0.0f; // zero pad filters
            firfilters[i * BLOCK_LEN + j][1] = 0.0f;
        }
    }
    taps.clear();

    fftwf_complex* lpfilter = (fftwf_complex*)malloc(PADDED_OLA_LENGTH * sizeof(fftwf_complex));
    for (int i = 0; i < PADDED_OLA_LENGTH; ++i)
    {
        if (i < NUM_TAPS)
            lpfilter[i][0] = taps2[i];            
        else
            lpfilter[i][0] = 0.0f; // zero pad filters            
        lpfilter[i][1] = 0.0f;
    }
    taps2.clear();
    //taps3.clear();

    // apply fft to filters
    data->firfiltersfft = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_FILTERS * sizeof(fftwf_complex));
    data->LP_filter = (fftwf_complex*)fftwf_malloc(OLA_FFT_OUTPUT_SIZE * sizeof(fftwf_complex));
    fftwf_plan filter_plans[NUM_FILTERS];
    fftwf_plan lp_filter_plan;
    for (int i = 0; i < NUM_FILTERS; ++i) // create the plans for calculating the fft of each filter block
    {
        filter_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &firfilters[i * BLOCK_LEN], &data->firfiltersfft[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE);
    }
    lp_filter_plan = fftwf_plan_dft_1d(PADDED_OLA_LENGTH, lpfilter, data->LP_filter, FFTW_FORWARD, FFTW_ESTIMATE);

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

    free(firfilters);
    free(lpfilter);
    printf("FIR filters are created.\n");

    printf("Setting up buffers.\n");
    float* theta = linspace(MIN_VIEW, NUM_VIEWS);
    float* phi = linspace(MIN_VIEW, NUM_VIEWS);
    float* delay = calcDelays(theta, phi);
    data->phase_shifts = calcPhaseShifts(delay);
    
    free(theta); free(phi); free(delay); // free memory which does not have to be allocated anymore*/    
    
    data->beams = (float*)malloc(NUM_VIEWS * NUM_VIEWS * NUM_FILTERS * sizeof(float));
    data->ordbuffer = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * sizeof(fftwf_complex));
    data->fft_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));
    data->filtered_data = (fftwf_complex*)fftwf_malloc(FFT_OUTPUT_SIZE * NUM_CHANNELS * NUM_FILTERS * sizeof(fftwf_complex));
    data->filtered_data_temp = (fftwf_complex*)fftwf_malloc(BLOCK_LEN * NUM_CHANNELS * NUM_FILTERS * sizeof(fftwf_complex));
    data->OLA_signal = (fftwf_complex*)fftwf_malloc(PADDED_OLA_LENGTH * NUM_CHANNELS * NUM_FILTERS * sizeof(fftwf_complex));
    //data->cosines = (float*)malloc(((NUM_OLA_BLOCK - 1) * FRAMES_PER_BUFFER + BLOCK_LEN) * NUM_FILTERS * sizeof(float));
    data->OLA_fft = (fftwf_complex*)fftwf_malloc(NUM_CHANNELS * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE * sizeof(fftwf_complex));
    data->OLA_decimated = (fftwf_complex*)malloc(DECIMATED_LENGTH * NUM_FILTERS * NUM_CHANNELS * sizeof(fftwf_complex));
    data->cosine_counter = (int*)malloc(NUM_CHANNELS * NUM_FILTERS * sizeof(int));
    data->summedSignals = (fftwf_complex*)malloc(DECIMATED_LENGTH * sizeof(fftwf_complex));    

    for (int i = 0; i < PADDED_OLA_LENGTH * NUM_CHANNELS * NUM_FILTERS; ++i)
    {
        data->OLA_signal[i][0] = 0.0f;
        data->OLA_signal[i][1] = 0.0f;
    }

    for (int i = 0; i < NUM_FILTERS * NUM_CHANNELS; ++i)
    {
        data->cosine_counter[i] = 0;
    }

    /*for (int i = 0; i < NUM_FILTERS; ++i)
    {
        float f_c = F_C + 2 * F_C * i; 
        for (int j = 0; j < OLA_LENGTH; ++j)
        {
            data->cosines[j + i * OLA_LENGTH] = cosf(2.0f * M_PI * f_c * (1.0f/SAMPLE_RATE) * j);
        }
    }*/

    printf("Creating fft plans.\n");
    for (int i = 0; i < NUM_CHANNELS; ++i) // create the plans for calculating the fft of each channel block
    {
        data->forw_plans[i] = fftwf_plan_dft_1d(BLOCK_LEN, &data->ordbuffer[i * BLOCK_LEN], &data->fft_data[i * FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE); // NUM_CHANNELS channels for each block which requires FFT_OUTPUT_SIZE spots to store the fft data
        for (int j = 0; j < NUM_FILTERS; ++j)
        {   // for each channel, there are NUM_FILTERS to apply, each application will need BLOCK_LEN spots in the array
            // inverse fft to get back the signal after filtering in freq-domain. each signal will need BLOCK_LEN spots, for each channel input there are NUM_FILTERS filters that have been applied to the input to form NUM_FILTERS nr of outputs
            data->back_plans[i * NUM_FILTERS + j] = fftwf_plan_dft_1d(BLOCK_LEN, &data->filtered_data[i * NUM_FILTERS * FFT_OUTPUT_SIZE + j * FFT_OUTPUT_SIZE], &data->filtered_data_temp[j * BLOCK_LEN + i * BLOCK_LEN * NUM_FILTERS], FFTW_BACKWARD, FFTW_ESTIMATE);
            data->OLA_forw[i * NUM_FILTERS + j] = fftwf_plan_dft_1d(PADDED_OLA_LENGTH, &data->OLA_signal[i * NUM_FILTERS * PADDED_OLA_LENGTH + j * PADDED_OLA_LENGTH], &data->OLA_fft[j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE], FFTW_FORWARD, FFTW_ESTIMATE);
            data->OLA_back[i * NUM_FILTERS + j] = fftwf_plan_dft_1d(PADDED_OLA_LENGTH, &data->OLA_fft[j * OLA_FFT_OUTPUT_SIZE + i * NUM_FILTERS * OLA_FFT_OUTPUT_SIZE], &data->OLA_signal[i * NUM_FILTERS * PADDED_OLA_LENGTH + j * PADDED_OLA_LENGTH], FFTW_BACKWARD, FFTW_ESTIMATE);
        }
    }

    // Create test signal, each channel will get the exact same signal for now
    fftwf_complex* input = (fftwf_complex*)malloc(FRAMES_PER_BUFFER * NUM_OLA_BLOCK * sizeof(fftwf_complex));
    for (int i = 0; i < FRAMES_PER_BUFFER * NUM_OLA_BLOCK; ++i)
    {
        input[i][0] = cosf(2 * M_PI * 520.0f * (1.0f / SAMPLE_RATE) * i);// + 0.f*cosf(2 * M_PI * 1700.0f * (1.0f / SAMPLE_RATE) * i) + \
                    cosf(2 * M_PI * 2750.0f * (1.0f / SAMPLE_RATE) * i) + cosf(2 * M_PI * 3400.0f * (1.0f / SAMPLE_RATE) * i);
        input[i][1] = 0.0f;
    }    

    // run the callback function 8 times
    for (int i = 0; i < NUM_OLA_BLOCK; ++i)
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
    checkErr(err, data);

    FILE* signal = popen("gnuplot", "w");*/

    std::vector<int> bins, ola_bins, time, time2;
    for (int i = 0; i < FFT_OUTPUT_SIZE; ++i)
    {
        bins.push_back(i);
    }

    for (int i = 0; i < BLOCK_LEN; ++i)
    {
        time.push_back(i);
    }

    for (int i = 0; i < OLA_LENGTH; ++i)
    {
        time2.push_back(i);
    }

    for (int i = 0; i < OLA_FFT_OUTPUT_SIZE; ++i)
    {
        ola_bins.push_back(i);
    }

    int v = 0; // which view you wanna look at for channel 1 (0 = channel 1, filter 1, 7 = channel 2 filter 1?)

    std::vector<float> ch1(FFT_OUTPUT_SIZE), lpf(FFT_OUTPUT_SIZE), res1(FFT_OUTPUT_SIZE), in(BLOCK_LEN), out(BLOCK_LEN), ola(OLA_LENGTH), LP(OLA_FFT_OUTPUT_SIZE), ola_fft(OLA_FFT_OUTPUT_SIZE);
    //while( ( err = Pa_IsStreamActive( stream ) ) == 1 )    
    //{
        for (int i = 0; i < FFT_OUTPUT_SIZE; ++i)
        {
            ch1.at(i) = sqrt(data->fft_data[i + FFT_OUTPUT_SIZE * v][0] * data->fft_data[i + FFT_OUTPUT_SIZE * v][0] + data->fft_data[i + FFT_OUTPUT_SIZE * v][1] * data->fft_data[i + FFT_OUTPUT_SIZE * v][1]);
            lpf.at(i) = sqrt(data->firfiltersfft[i + FFT_OUTPUT_SIZE * v][0] * data->firfiltersfft[i + FFT_OUTPUT_SIZE * v][0] + data->firfiltersfft[i + FFT_OUTPUT_SIZE * v][1] * data->firfiltersfft[i + FFT_OUTPUT_SIZE * v][1]);
            res1.at(i) = sqrt(data->filtered_data[i + FFT_OUTPUT_SIZE * v][0] * data->filtered_data[i + FFT_OUTPUT_SIZE * v][0] + data->filtered_data[i + FFT_OUTPUT_SIZE * v][1] * data->filtered_data[i + FFT_OUTPUT_SIZE * v][1]);
            //printf("Block i: %f\n", data->filtered_data_temp[i][1]);
        }

        for (int i = 0; i < BLOCK_LEN; ++i)
        {
            in.at(i) = data->ordbuffer[i + BLOCK_LEN * v][0];
            out.at(i) = data->filtered_data_temp[i + BLOCK_LEN * v][0] / BLOCK_LEN;
        }

        for (int i = 0; i < OLA_LENGTH; ++i)
        {
            ola.at(i) = data->OLA_signal[i + OLA_LENGTH * v][0];            
        }

        for (int i = 0; i < OLA_FFT_OUTPUT_SIZE; ++i)
        {
            ola_fft.at(i) = sqrt(data->OLA_fft[i + OLA_FFT_OUTPUT_SIZE * v][0] * data->OLA_fft[i + OLA_FFT_OUTPUT_SIZE * v][0] + data->OLA_fft[i + OLA_FFT_OUTPUT_SIZE * v][1] * data->OLA_fft[i + OLA_FFT_OUTPUT_SIZE * v][1]);
            LP.at(i) = sqrt(data->LP_filter[i][0] * data->LP_filter[i][0] + data->LP_filter[i][1] * data->LP_filter[i][1]);
        }

        plt::figure(10);
        plt::title("Frequency contents, channel 1");
        plt::clf();    
        plt::plot(bins, ch1);
        plt::xlabel("freq bin");

        plt::figure(11);
        plt::title("Frequency contents, lp filter");
        plt::clf();    
        plt::plot(bins, lpf);
        plt::xlabel("freq bin");

        /*plt::figure(12);
        plt::title("Frequency contents, filter*signal");
        plt::clf();    
        plt::plot(bins, res1);
        plt::xlabel("freq bin");*/

        plt::figure(13);
        plt::title("In signal");
        plt::clf();    
        plt::plot(time, in);
        plt::xlabel("time");

        plt::figure(14);
        plt::title("Out signal");
        plt::clf();    
        plt::plot(time, out);
        plt::xlabel("time");

        plt::figure(15);
        plt::title("OLA signal");
        plt::clf();    
        plt::plot(time2, ola);
        plt::xlabel("time");

        plt::figure(16);
        plt::title("OLA_fft");
        plt::clf();    
        plt::plot(ola_bins, ola_fft);
        plt::xlabel("freq bin");

        /*plt::figure(17);
        plt::title("LP_filter");
        plt::clf();    
        plt::plot(ola_bins, LP);
        plt::xlabel("freq bin");*/

        //plt::pause(2.0);
        plt::show();

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

fftwf_complex* calcPhaseShifts(float* delay){
    // for each lobe (NUM_VIEWS x NUM_VIEWS), there are NUM_CHANNELS * NUM_FILTERS signals to process
    // first up is channel 1, filter 1, e^(-jwt_0), then channel 1, filter 1, e^(jw(t_0 + Ts)), ..., channel 1, filter 1 e^(jw(t_0 + Ts * (N-1))), (N is number of samples the decimated OLA signal contains)
    // then comes channel 1, filter 2, e^(-jwt_0), ...
    // ...
    // for channel 2, filter 1 it becomes e^(-jwt_1) (t_0, t_1, ..., t_(M-1) are given by delay)
    // w is dependent on the filter, w is the original center frequency of the filter applied to the signal f_c = F_C + 2 * F_C * i; 
    //     
    fftwf_complex* ps = (fftwf_complex*)malloc(NUM_CHANNELS * NUM_FILTERS * NUM_VIEWS * NUM_VIEWS * DECIMATED_LENGTH * sizeof(fftwf_complex));

    float f_c, omega, tau;    
    for (int i = 0; i < NUM_VIEWS * NUM_VIEWS; ++i)
    {
        for (int j = 0; j < NUM_CHANNELS; ++j)
        {
            for (int k = 0; k < NUM_FILTERS; ++k)
            {
                f_c = F_C + k * 2 * F_C;
                omega = 2.0f * M_PI * f_c;
                tau = delay[j + i * NUM_CHANNELS]; // for a specific view, the delay is x for channel j

                for (int l = 0; l < DECIMATED_LENGTH; ++l)
                {
                    ps[l + k * DECIMATED_LENGTH + j * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][0] = \
                    cosf(-omega * (tau + l * (1.0f/SAMPLE_RATE) * DECIMATED_STEP));

                    ps[l + k * DECIMATED_LENGTH + j * NUM_FILTERS * DECIMATED_LENGTH + i * NUM_CHANNELS * NUM_FILTERS * DECIMATED_LENGTH][1] = \
                    sinf(-omega * (tau + l * (1.0f/SAMPLE_RATE) * DECIMATED_STEP));
                }
            }        
        }        
    }
}

/*int* calca(float* delay)
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
}*/