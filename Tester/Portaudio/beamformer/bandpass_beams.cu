#include "bandpass_beams.h"

__global__
void bandpass_filtering_calcs(int i, cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter)
{
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index, from 0 to BLOCK_LEN - 1
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN; // internal index + compensation for which beam is being calced
    int l3 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN * NUM_FILTERS; // as l2, but compensates for beams being calced in different freq-bands
    //       -           0 - 2047               -, + i *  2048     *     6

    for (int j = 0; j < NUM_FILTERS; ++j)
    {        
        summedSignals_fft_BP[l3 + j * BLOCK_LEN].x = summedSignals_fft[l2].x * BP_filter[l1 + j * BLOCK_LEN].x - summedSignals_fft[l2].y * BP_filter[l1 + j * BLOCK_LEN].y;
        summedSignals_fft_BP[l3 + j * BLOCK_LEN].y = summedSignals_fft[l2].x * BP_filter[l1 + j * BLOCK_LEN].y + summedSignals_fft[l2].y * BP_filter[l1 + j * BLOCK_LEN].x;
        // 0     - 2047  : beam 1, filter 1
        // 2048  - 4095  : beam 1, filter 2
        // 12228 - 14335 : beam 2, filter 1 
    }
    // after these calculations there should be NUM_FILTERS signals per view, and each signals is BLOCK_LEN samples long, the strength of the signals need to be calced
}

__global__
void bandpass_filtering(cufftComplex* summedSignals_fft_BP, cufftComplex* summedSignals_fft, cufftComplex* BP_filter, float* beams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;    

    if (i >= NUM_BEAMS * NUM_BEAMS){
        return;
    }

    // calculations
    bandpass_filtering_calcs<<<(BLOCK_LEN+255)/256.0f, 256>>>(i, summedSignals_fft_BP, summedSignals_fft, BP_filter);
    cudaDeviceSynchronize();

    float beamstrength;
    int id;
    for (int j = 0; j < NUM_FILTERS; ++j)
    {
        beamstrength = 0.0f;
        for (int k = 0; k < BLOCK_LEN; ++k)
        {
            id = k + j * BLOCK_LEN + i * NUM_FILTERS * BLOCK_LEN;

            beamstrength += summedSignals_fft_BP[id].x * summedSignals_fft_BP[id].x + summedSignals_fft_BP[id].y * summedSignals_fft_BP[id].y;
        }
        beams[i + j * NUM_BEAMS * NUM_BEAMS] = 20.0f * log10(sqrtf(beamstrength) / ( (float)NUM_CHANNELS * (float)(BLOCK_LEN * BLOCK_LEN * sqrtf((float)BLOCK_LEN))));
    }
}