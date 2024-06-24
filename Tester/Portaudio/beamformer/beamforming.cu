#include "beamforming.h"

__global__
void interpolateChannels(const cufftComplex* inputBuffer, cufftComplex* summedSignals, const int i, const int* a, const int* b, const float* alpha, const float* beta)
{
    int id;    
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index of this thread
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN; // global index of this thread

    // l1 -> 0 - 2047
    // l2 -> 0 - 2047 + i * 2048, i -> 0 - 168

    summedSignals[l2].x = 0.0f;
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
            summedSignals[l2].x += beta[id] * inputBuffer[l1 + b[id] + k * BLOCK_LEN].x; // do not write to the first b[id]-1 positions
    }    
}

__global__ 
void beamforming(const cufftComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftComplex* summedSignals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_BEAMS * NUM_BEAMS){
        return;
    }

    // interpolate channels    
    interpolateChannels<<<(BLOCK_LEN+255)/256, 256>>>(inputBuffer, summedSignals, i, a, b, alpha, beta);
    cudaDeviceSynchronize();    
}