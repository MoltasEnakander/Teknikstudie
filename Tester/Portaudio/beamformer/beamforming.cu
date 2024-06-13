#include "beamforming.h"

/*__global__
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
}*/

__global__
void interpolateChannels(const float* time_stamps, float dt, cufftComplex* summedSignals, const int i, \
                            const float* coeff1, const float* coeff2, const float* coeff3, const float* coeff4, float* mus, float* mus2, float* mus3)
{
    // i indicates the current beam    
    int l1 = blockIdx.x * blockDim.x + threadIdx.x; // internal index of this thread
    int l2 = blockIdx.x * blockDim.x + threadIdx.x + i * BLOCK_LEN; // global index of this thread

    // l1 -> 0 - 2047
    // l2 -> 0 - 2047 + i * 2048, i -> 0 - 168

    std::size_t knot;
    float mu, mu2, mu3;

    float dt_inv = 1.0f / dt;

    summedSignals[l2].x = 0.0f;
    for (int j = 0; j < NUM_CHANNELS; ++j)
    {
        mu = time_stamps[i * NUM_CHANNELS + j] + (float)(l1 * dt);
        //   delay for beam i, channel k       + sample * delta_T
        // mu is the time point where interpolation should happen

        knot = (std::size_t)(mu * dt_inv + 1.0e-15);
        // knot is the sample where interpolation should happen

        if (knot >= BLOCK_LEN) // sample to interpolate with has not reached the microphone yet
            continue;

        mu = mu - (float)knot*dt;
        // mu is now the time difference between the time_stamp where interpolation
        // should happen and the time_stamp that corresponds to sample knot

        mu2 = mu * mu;
        mu3 = mu2 * mu;

        mus[l1 + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] = mu;
        mus2[l1 + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] = mu2;
        mus3[l1 + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] = mu3;

        summedSignals[l2].x += coeff1[knot + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] + \
                                coeff2[knot + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] * mu + \
                                coeff3[knot + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] * mu2 + \
                                coeff4[knot + j * BLOCK_LEN + i * NUM_CHANNELS * BLOCK_LEN] * mu3;
    }    
}

//void beamforming(const cufftComplex* inputBuffer, const int* a, const int* b, float* alpha, const float* beta, cufftComplex* summedSignals)
__global__ 
void beamforming(const float* delays, const float* coeff1, const float* coeff2, const float* coeff3, const float* coeff4, cufftComplex* summedSignals, float* mus, float* mus2, float* mus3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_BEAMS * NUM_BEAMS){
        return;
    }

    // interpolate channels    
    //interpolateChannels<<<(BLOCK_LEN+255)/256, 256>>>(inputBuffer, summedSignals, i, a, b, alpha, beta);
    interpolateChannels<<<(BLOCK_LEN+255)/256.0f, 256>>>(delays, 1.0f/SAMPLE_RATE, summedSignals, i, coeff1, coeff2, coeff3, coeff4, mus, mus2, mus3);
    cudaDeviceSynchronize();    
}