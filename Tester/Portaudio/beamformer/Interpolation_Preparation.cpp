#include "Interpolation_Preparation.h"
#include <stdio.h>

float* linspace(int a, int num)
{
    float* f = (float*)malloc(NUM_BEAMS*sizeof(float));    
             
    // now assign the values to the array
    for (int i = 0; i < num; i++)
    {
        f[i] = (a + i * VIEW_INTERVAL) * M_PI / 180.0f;
    }
    return f;
}

float* calcDelays(float* theta, float* phi)
{
    float* d = (float*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(float));    

    int pid = 0; // phi index
    int tid = 0; // theta index
    for (int i = 0; i < NUM_BEAMS * NUM_BEAMS; ++i){
        for (int k = 0; k < NUM_CHANNELS; ++k){
            d[k + i * NUM_CHANNELS] = -(xa[k] * sinf(theta[tid]) * cosf(phi[pid]) + ya[k] * sinf(phi[pid])) * ARRAY_DIST / C * SAMPLE_RATE;
        }

        tid++;
        if (tid >= NUM_BEAMS){
            tid = 0;
            pid++;
        }
    }
    return d;
}

int* calca(float* delay)
{
    int* a = (int*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        a[i] = floor(delay[i]);
    }
    return a;
}

int* calcb(int* a)
{
    int* b = (int*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(int));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        b[i] = a[i] + 1;
    }
    return b;
}

float* calcalpha(float* delay, int* b)
{
    float* alpha = (float*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        alpha[i] = b[i] - delay[i];
    }
    return alpha;
}

float* calcbeta(float* alpha)
{
    float* beta = (float*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(float));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        beta[i] = 1 - alpha[i];
    }
    return beta;
}