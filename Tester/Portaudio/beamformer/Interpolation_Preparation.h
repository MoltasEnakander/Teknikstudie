#ifndef INTERPOLATION_PREPARATION_H
#define INTERPOLATION_PREPARATION_H
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include "definitions.h"

// positions in the microphone array
static float xa[16] = {-0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, -0.5f, -1.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f, 1.5f, 0.5f};
static float ya[16] = {-1.5f, -1.5f, -0.5f, -0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 1.5f, 0.5f, 0.5f, -0.5f, -0.5f, -1.5f, -1.5f};

float* linspace(int a, int num);

float* calcDelays(float* theta, float* phi);

int* calca(float* delay);

int* calcb(int* a);

float* calcalpha(float* delay, int* b);

float* calcbeta(float* alpha);

#endif