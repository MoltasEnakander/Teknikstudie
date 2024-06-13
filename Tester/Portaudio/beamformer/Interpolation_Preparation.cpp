#include "Interpolation_Preparation.h"
#include <stdio.h>

double* linspace(int a, int num)
{
    // create a vector of length num
    //std::vector<double> v(NUM_BEAMS, 0);    
    double* f = (double*)malloc(NUM_BEAMS*sizeof(double));    
             
    // now assign the values to the array
    for (int i = 0; i < num; i++)
    {
        f[i] = (a + i * VIEW_INTERVAL) * M_PI / 180.0f;
    }
    return f;
}

double* calcDelays(double* theta, double* phi)
{
    double* d = (double*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(double));    

    int pid = 0; // phi index
    int tid = 0; // theta index
    for (int i = 0; i < NUM_BEAMS * NUM_BEAMS; ++i){
        double min = 1e10;
        for (int k = 0; k < NUM_CHANNELS; ++k){
            d[k + i * NUM_CHANNELS] = -(xa[k] * sinf(theta[tid]) * cosf(phi[pid]) + ya[k] * sinf(phi[pid])) * ARRAY_DIST / C;// * SAMPLE_RATE;
            if (d[k + i * NUM_CHANNELS] < min)
                min = d[k + i * NUM_CHANNELS];
        }
        for (int k = 0; k < NUM_CHANNELS; ++k)
        {
            d[k + i * NUM_CHANNELS] -= min;
            //printf("Delay for beam %d, channel %d: %f\n", i + 1, k + 1, d[k + i * NUM_CHANNELS]);
        }

        tid++;
        if (tid >= NUM_BEAMS){
            tid = 0;
            pid++;
        }
    }
    return d;
}

int* calca(double* delay)
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

double* calcalpha(double* delay, int* b)
{
    double* alpha = (double*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(double));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        alpha[i] = b[i] - delay[i];
    }
    return alpha;
}

double* calcbeta(double* alpha)
{
    double* beta = (double*)malloc(NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS*sizeof(double));
    for (int i = 0; i < NUM_BEAMS*NUM_BEAMS*NUM_CHANNELS; ++i)
    {
        beta[i] = 1 - alpha[i];
    }
    return beta;
}