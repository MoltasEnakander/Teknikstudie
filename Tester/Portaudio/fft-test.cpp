#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cufft.h>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "fftw3.h"
#include <chrono>
#include <ctime>
#include <unistd.h>

int main()
{
	const int FRAMES_PER_BUFFER = 2048;
	const int NUM_CHANNELS = 16;
	//const int NUM_ELEMENTS = FRAMES_PER_BUFFER * NUM_CHANNELS;
    const int SINGLE_OUTPUT_SIZE = FRAMES_PER_BUFFER / 2 + 1;
	//const int OUTPUT_SIZE = SINGLE_OUTPUT_SIZE * NUM_CHANNELS;
	const float Fs = 44100.0f;
	const float A = 5.0f;
	const float f = 50.0f;
	float xt[NUM_CHANNELS][FRAMES_PER_BUFFER];
	float signal[FRAMES_PER_BUFFER*NUM_CHANNELS];
	float ordsignal[FRAMES_PER_BUFFER*NUM_CHANNELS];

	cufftHandle plan1d, planMany;
	cufftComplex *data1d, *dataMany, *h_fft1d, *h_fftMany;
    cufftReal *d_signal, *d_ordsignal;
    
    fftwf_complex *fft_cpu;
    float *fftwf_input;
    float *fftwf_output;
    //fftwf_plan p1, p2, p3, p4;
    fftwf_plan plans[NUM_CHANNELS];

    fftwf_plan bplans[NUM_CHANNELS];

	// create sine signals, each channel will have a different pure sine wave
	for (int i = 0; i < NUM_CHANNELS; ++i)
	{
		for (int j = 0; j < FRAMES_PER_BUFFER; ++j)
		{			
			xt[i][j] = A * sinf(2.0f * M_PI * f * (i+1) * (1.0f/Fs) * j);
		}
	}	
	
	// copy to 1d array
	for (int i = 0; i < NUM_CHANNELS*FRAMES_PER_BUFFER; ++i)
	{
		int id = i % NUM_CHANNELS;
		int id2 = i / NUM_CHANNELS;
		signal[i] = xt[id][id2];		
	}

	// copy to another 1d array which is sorted by channel
	for (int i = 0; i < NUM_CHANNELS; ++i)
    {       
        for (int j = 0; j < FRAMES_PER_BUFFER; ++j)
        {                       
            ordsignal[i * FRAMES_PER_BUFFER + j] = signal[j * NUM_CHANNELS + i];
        }        
    }

    // create time-axis and signal vectors (for visualization purposes)
    std::vector<float> time, s1, s2; 

    for (int i = 0; i < FRAMES_PER_BUFFER; ++i)
    {
    	time.push_back(i * 1.0f / Fs);
    	s1.push_back(xt[0][i]);
        s2.push_back(ordsignal[i]);
    }

    // plot signal(s)
    /*plt::figure(1);
	plt::title("Time signal");
    plt::clf();
    plt::plot(time, s2);
    plt::xlabel("time (s)");*/

    // allocate GPU memory
    /*cudaMalloc((void **)&data1d, sizeof(cufftComplex)*OUTPUT_SIZE);
    cudaMalloc((void **)&dataMany, sizeof(cufftComplex)*OUTPUT_SIZE);
    cudaMalloc((void **)&d_signal, sizeof(cufftReal)*NUM_ELEMENTS);
    cudaMalloc((void **)&d_ordsignal, sizeof(cufftReal)*NUM_ELEMENTS);

    // copy data to GPU
    cudaMemcpy(d_signal, signal, NUM_ELEMENTS*sizeof(cufftReal), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ordsignal, ordsignal, NUM_ELEMENTS*sizeof(cufftReal), cudaMemcpyHostToDevice);

    // create the plans
    int n[1] = {FRAMES_PER_BUFFER};
    int inembed[] = {FRAMES_PER_BUFFER};
    int onembed[] = {SINGLE_OUTPUT_SIZE};

	cufftPlan1d(&(plan1d), FRAMES_PER_BUFFER, CUFFT_R2C, NUM_CHANNELS);
    cufftPlanMany(&(planMany), 1, n, inembed, NUM_CHANNELS, 1, onembed, NUM_CHANNELS, 1, CUFFT_R2C, NUM_CHANNELS);*/

    fftwf_input = (float*)fftwf_malloc(FRAMES_PER_BUFFER * NUM_CHANNELS * sizeof(float));
    fftwf_output = (float*)fftwf_malloc((FRAMES_PER_BUFFER + 128) * NUM_CHANNELS * sizeof(float));

    memcpy(fftwf_input, ordsignal, FRAMES_PER_BUFFER * NUM_CHANNELS * sizeof(float));

    

    fft_cpu = (fftwf_complex*)fftwf_malloc(SINGLE_OUTPUT_SIZE * NUM_CHANNELS * sizeof(fftwf_complex));

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        plans[i] = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, &fftwf_input[i * FRAMES_PER_BUFFER], &fft_cpu[i * SINGLE_OUTPUT_SIZE], FFTW_ESTIMATE);
    }

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        bplans[i] = fftwf_plan_dft_c2r_1d(FRAMES_PER_BUFFER, &fft_cpu[i * SINGLE_OUTPUT_SIZE], &fftwf_output[i * (FRAMES_PER_BUFFER+128)], FFTW_ESTIMATE);
    }

	// run the fft-calculations    

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();    

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_execute(plans[i]);
        
    }    

    std::vector<float> v_fft_in; 
    for (int i = 0; i < FRAMES_PER_BUFFER; ++i)
    {
        v_fft_in.push_back(fftwf_input[i]);
    }

    plt::figure(2);
    plt::title("Time signal");
    plt::clf();
    plt::plot(time, v_fft_in);
    plt::xlabel("time (s)");
        
    
    /*start = std::chrono::system_clock::now();

    for (int i = 0; i < 1; ++i)
    {
        cufftExecR2C(plan1d, d_ordsignal, data1d);
        cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
    elapsed = end-start;
    std::cout << "elapsed: " << elapsed.count() << "s\n";*/
    // ------------------------------------------------------
    /*start = std::chrono::system_clock::now();
    
    for (int i = 0; i < 1; ++i)
    {        
        cufftExecR2C(planMany, d_signal, dataMany);
        cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
    elapsed = end-start;
    std::cout << "elapsed: " << elapsed.count() << "s\n";*/

    // copy data back to CPU    
    /*h_fft1d = (cufftComplex*)malloc(sizeof(cufftComplex)*OUTPUT_SIZE);
    h_fftMany = (cufftComplex*)malloc(sizeof(cufftComplex)*OUTPUT_SIZE);
    cudaMemcpy(h_fft1d, data1d, OUTPUT_SIZE*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fftMany, dataMany, OUTPUT_SIZE*sizeof(cufftComplex), cudaMemcpyDeviceToHost);*/
    
    // create freq bins vector and fft-signal vector(s)
    std::vector<int> bins;
    std::vector<float> c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24;
    for (int i = 0; i < SINGLE_OUTPUT_SIZE; ++i)
    {
    	bins.push_back(i);

    	// plan1d data
    	c1.push_back(sqrt(fft_cpu[i][0] * fft_cpu[i][0] + fft_cpu[i][1] * fft_cpu[i][1]));
        //printf("%d: (%f, %f) - abs = %f \n", i, fft_cpu[i][0], fft_cpu[i][1], sqrt(fft_cpu[i][0] * fft_cpu[i][0] + fft_cpu[i][1] * fft_cpu[i][1]) );
    	/*c2.push_back(sqrt(h_fft1d[i + SINGLE_OUTPUT_SIZE].x * h_fft1d[i + SINGLE_OUTPUT_SIZE].x + h_fft1d[i + SINGLE_OUTPUT_SIZE].y * h_fft1d[i + SINGLE_OUTPUT_SIZE].y));
    	c3.push_back(sqrt(h_fft1d[i + SINGLE_OUTPUT_SIZE * 2].x * h_fft1d[i + SINGLE_OUTPUT_SIZE * 2].x + h_fft1d[i + SINGLE_OUTPUT_SIZE * 2].y * h_fft1d[i + SINGLE_OUTPUT_SIZE * 2].y));
    	c4.push_back(sqrt(h_fft1d[i + SINGLE_OUTPUT_SIZE * 3].x * h_fft1d[i + SINGLE_OUTPUT_SIZE * 3].x + h_fft1d[i + SINGLE_OUTPUT_SIZE * 3].y * h_fft1d[i + SINGLE_OUTPUT_SIZE * 3].y));*/
    	/*c5.push_back(ordsignal[i + FRAMES_PER_BUFFER * 4]);
    	c6.push_back(ordsignal[i + FRAMES_PER_BUFFER * 5]);
    	c7.push_back(ordsignal[i + FRAMES_PER_BUFFER * 6]);
    	c8.push_back(ordsignal[i + FRAMES_PER_BUFFER * 7]);*/

    	// planMany data
    	//c9.push_back(sqrt(h_fftMany[i*NUM_CHANNELS].x * h_fftMany[i*NUM_CHANNELS].x + h_fftMany[i*NUM_CHANNELS].y * h_fftMany[i*NUM_CHANNELS].y));
    	/*c10.push_back(signal[i * NUM_CHANNELS + 1]);
    	c11.push_back(signal[i * NUM_CHANNELS + 2]);
    	c12.push_back(signal[i * NUM_CHANNELS + 3]);
    	c13.push_back(signal[i * NUM_CHANNELS + 4]);
    	c14.push_back(signal[i * NUM_CHANNELS + 5]);
    	c15.push_back(signal[i * NUM_CHANNELS + 6]);
    	c16.push_back(signal[i * NUM_CHANNELS + 7]);*/

        /*c17.push_back(sqrt(fft_cpu[i][0] * fft_cpu[i][0] + fft_cpu[i][1] * fft_cpu[i][1]));
        c18.push_back(sqrt(fft_cpu[i + SINGLE_OUTPUT_SIZE][0] * fft_cpu[i + SINGLE_OUTPUT_SIZE][0] + fft_cpu[i + SINGLE_OUTPUT_SIZE][1] * fft_cpu[i + SINGLE_OUTPUT_SIZE][1]));
        c19.push_back(sqrt(fft_cpu[i + 2*SINGLE_OUTPUT_SIZE][0] * fft_cpu[i + 2*SINGLE_OUTPUT_SIZE][0] + fft_cpu[i + 2*SINGLE_OUTPUT_SIZE][1] * fft_cpu[i + 2*SINGLE_OUTPUT_SIZE][1]));
        c20.push_back(sqrt(fft_cpu[i + 3*SINGLE_OUTPUT_SIZE][0] * fft_cpu[i + 3*SINGLE_OUTPUT_SIZE][0] + fft_cpu[i + 3*SINGLE_OUTPUT_SIZE][1] * fft_cpu[i + 3*SINGLE_OUTPUT_SIZE][1]));*/
    }

    

    plt::figure(9);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(bins, c1);
    plt::xlabel("bins");

    printf("INvers\n");
    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_execute(bplans[i]);        
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end-start;
    std::cout << "elapsed: " << elapsed.count() << "s\n";

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        for (int j = 0; j < 128; ++j)
        {
            fftwf_output[(FRAMES_PER_BUFFER + 128) * i + j + FRAMES_PER_BUFFER] = 2048.0f;
        }
    }

    int BLOCK_LEN = FRAMES_PER_BUFFER + 128;
    std::vector<float> s3, s32, time2; 
    for (int i = 0; i < BLOCK_LEN; ++i)
    {        
        s3.push_back(fftwf_output[i] / FRAMES_PER_BUFFER);
        s32.push_back(fftwf_output[i + BLOCK_LEN] / FRAMES_PER_BUFFER);
        time2.push_back(i);
    }

    // plot signal(s)
    plt::figure(32);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(time2, s3);
    plt::xlabel("time (s)");
    

    plt::figure(322);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(time2, s32);
    plt::xlabel("time (s)");

    int OLA_block = 6;
    //int NUM_CHANNELS = 16;
    int NUM_FILTERS = 1; // needs to be 1
    
    float* OLA = (float*)fftwf_malloc(((OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN) * NUM_CHANNELS * NUM_FILTERS * sizeof(float));

    for (int i = 0; i < ((OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN) * NUM_CHANNELS * NUM_FILTERS; ++i)
    {
        OLA[i] = 0.0f;
    }

    

    // shift once
    /*for (int k = 0; k < OLA_block - 1; ++k)
    {
        // shift FRAMES_PER_BUFFER to the left
        for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
        {
            OLA[l + k * FRAMES_PER_BUFFER] = OLA[l + (k+1) * FRAMES_PER_BUFFER];
        }
    }
    // shift in last part and add new part
    for (int l = 0; l < 128; ++l)
    {
        OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER] = OLA[l + OLA_block * FRAMES_PER_BUFFER] + fftwf_output[l] / FRAMES_PER_BUFFER;
    }
    for (int l = BLOCK_LEN - FRAMES_PER_BUFFER; l < BLOCK_LEN; ++l)
    {
        OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER] = fftwf_output[l] / FRAMES_PER_BUFFER;
    }*/

    printf("Dags för första skift\n");

    int single_OLA_space = (OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN;
    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        for (int j = 0; j < NUM_FILTERS; ++j)
        {
            for (int k = 0; k < OLA_block - 1; ++k)
            {
                // shift FRAMES_PER_BUFFER to the left
                for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
                {                    
                    OLA[l + k * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                    OLA[l + (k+1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space];
                }
            }            
            // shift in last part and add new part
            for (int l = 0; l < BLOCK_LEN - FRAMES_PER_BUFFER; ++l)
            {
                OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                OLA[l + OLA_block * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] + \
                fftwf_output[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN] / BLOCK_LEN;
            }
            for (int l = BLOCK_LEN - FRAMES_PER_BUFFER; l < BLOCK_LEN; ++l)
            {
                OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                fftwf_output[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN] / BLOCK_LEN;
            }
        }
    }

    std::vector<float> s4, s42, timesg; 
    for (int i = 0; i < (OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN; ++i)
    {        
        s4.push_back(OLA[i]);
        s42.push_back(OLA[i + (OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN]);
        timesg.push_back(i);
    }

    plt::figure(33);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(timesg, s4);
    plt::xlabel("time (s)");

    plt::figure(332);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(timesg, s42);
    plt::xlabel("time (s)");

    printf("Dags för andra skift\n");
    // shift again
    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        for (int j = 0; j < NUM_FILTERS; ++j)
        {
            for (int k = 0; k < OLA_block - 1; ++k)
            {
                // shift FRAMES_PER_BUFFER to the left
                for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
                {                    
                    OLA[l + k * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                    OLA[l + (k+1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space];
                }
            }            
            // shift in last part and add new part
            for (int l = 0; l < BLOCK_LEN - FRAMES_PER_BUFFER; ++l)
            {
                OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                OLA[l + OLA_block * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] + \
                fftwf_output[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN] / BLOCK_LEN;
            }
            for (int l = BLOCK_LEN - FRAMES_PER_BUFFER; l < BLOCK_LEN; ++l)
            {
                OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER + j * single_OLA_space + i * NUM_FILTERS * single_OLA_space] = \
                fftwf_output[l + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN] / BLOCK_LEN;
            }
        }
    }
    /*for (int k = 0; k < OLA_block - 1; ++k)
    {
        // shift FRAMES_PER_BUFFER to the left
        for (int l = 0; l < FRAMES_PER_BUFFER; ++l)
        {
            OLA[l + k * FRAMES_PER_BUFFER] = OLA[l + (k+1) * FRAMES_PER_BUFFER];
        }
    }
    // shift in last part and add new part
    for (int l = 0; l < 128; ++l)
    {
        OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER] = OLA[l + OLA_block * FRAMES_PER_BUFFER] + fftwf_output[l] / FRAMES_PER_BUFFER;
    }
    for (int l = BLOCK_LEN - FRAMES_PER_BUFFER; l < BLOCK_LEN; ++l)
    {
        OLA[l + (OLA_block - 1) * FRAMES_PER_BUFFER] = fftwf_output[l] / FRAMES_PER_BUFFER;
    }*/


    std::vector<float> s5, s52;
    for (int i = 0; i < (OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN; ++i)
    {        
        s5.push_back(OLA[i]); 
        s52.push_back(OLA[i + (OLA_block - 1) * FRAMES_PER_BUFFER + BLOCK_LEN]);
    }

    plt::figure(34);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(timesg, s5);
    plt::xlabel("time (s)");

    plt::figure(342);
    plt::title("Time signal after fft");
    plt::clf();
    plt::plot(timesg, s52);
    plt::xlabel("time (s)");

    // plot frequency contents of channels
    /*plt::figure(2);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c1);
    plt::xlabel("freq bin");

    plt::figure(3);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c2);
    plt::xlabel("freq bin");

    plt::figure(4);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c3);
    plt::xlabel("freq bin");

    plt::figure(5);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c4);
    plt::xlabel("freq bin");

    plt::figure(6);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c9);
    plt::xlabel("freq bin");*/

    /*plt::figure(7);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c17);
    plt::xlabel("freq bin");

    plt::figure(8);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c18);
    plt::xlabel("freq bin");

    plt::figure(9);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c19);
    plt::xlabel("freq bin");

    plt::figure(10);
    plt::title("Frequency contents");
    plt::clf();    
    plt::plot(bins, c20);
    plt::xlabel("freq bin");*/

    plt::show();

    // free data
    cufftDestroy(plan1d);
    cufftDestroy(planMany);
    cudaFree(data1d);
    cudaFree(dataMany);
    cudaFree(d_signal);
    cudaFree(d_ordsignal);
    free(h_fft1d);
    free(h_fftMany);
    fftwf_free(fft_cpu);
    fftwf_free(fftwf_input);
    fftwf_free(fftwf_output);

    for (int i = 0; i < NUM_CHANNELS; ++i)
    {
        fftwf_destroy_plan(plans[i]);
        fftwf_destroy_plan(bplans[i]);
    }

    /*fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);
    fftwf_destroy_plan(p3);
    fftwf_destroy_plan(p4);*/

	return 0;
}