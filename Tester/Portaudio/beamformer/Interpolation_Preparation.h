#ifndef INTERPOLATION_PREPARATION_H
#define INTERPOLATION_PREPARATION_H
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include "definitions.h"

// positions in the microphone array
static double xa[16] = {-0.5, -1.5, -0.5, -1.5, -0.5, -1.5, -0.5, -1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5};
static double ya[16] = {-1.5, -1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, -0.5, -0.5, -1.5, -1.5};

double* linspace(int a, int num);

double* calcDelays(double* theta, double* phi);

int* calca(double* delay);

int* calcb(int* a);

double* calcalpha(double* delay, int* b);

double* calcbeta(double* alpha);

#endif