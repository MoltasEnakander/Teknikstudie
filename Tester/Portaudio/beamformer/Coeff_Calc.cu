#include "Coeff_Calc.h"

__global__
void spline_init(cufftDoubleComplex* y, double dx, std::size_t signal_length, double* coeff1, double* coeff2, double* coeff3, double* coeff4, double* cp, double* dp, double* d, double* sigma)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_BEAMS * NUM_BEAMS * NUM_CHANNELS){
        return;
    }

    int k = i / NUM_CHANNELS; // k now denotes the beam
    int j = i % NUM_CHANNELS; // j denotes the channel of the current beam

    /*
	i = 0 -> k = 0, j = 0, beam 1, channel 1
	i = 10 -> k = 0, j = 10, beam 1, channel 11
	i = 49 -> k = 3, j = 1, beam 4, channel 2
    */

    //Based on http://www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture11.pdf
	double h = dx;
	double h2 = h * h;
	double h_inv = 1.0f / h;
	double h2_inv = 1.0f / h2;
	const double c_1_6 = 1.0f / 6.0f;
	const int n = int(signal_length);

	int global_id = j * BLOCK_LEN + k * NUM_CHANNELS * BLOCK_LEN;

	sigma[0 + global_id] = 0.0f;
	sigma[n - 1 + global_id] = 0.0f;
	for (int idx = 1; idx < n - 1; ++idx)
	{		
		d[idx + global_id] = 6.0f * h2_inv * (y[idx + 1 + j * BLOCK_LEN].x - 2.0f * y[idx + j * BLOCK_LEN].x + y[idx - 1 + j * BLOCK_LEN].x);
	}

	//Using Tridiagonal matrix algorithm
	//Thomas algorithm (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)

	double a_0 = 1.0f;
	double b_0 = 4.0f;
	double c_0 = 1.0f;
	cp[1 + global_id] = c_0 / b_0;
	dp[1 + global_id] = d[1 + global_id]/b_0;


	for (int idx = 2; idx < n - 1; ++idx)
	{
		cp[idx + global_id] = c_0 / (b_0 - a_0 * cp[idx - 1 + global_id]);
		dp[idx + global_id] = (d[idx + global_id] - a_0 * dp[idx - 1 + global_id]) / (b_0 - a_0 * cp[idx - 1 + global_id]);
	}
	sigma[n - 2 + global_id] = dp[n - 2 + global_id];

	for (int idx = n - 3; idx > 0; --idx)
	{
		sigma[idx + global_id] = dp[idx + global_id] - cp[idx + global_id] * sigma[idx + 1 + global_id];
	}
	for (int idx = 0; idx < n - 1; ++idx)
	{
		coeff1[idx + global_id] = y[idx + j * BLOCK_LEN].x; // d
		coeff3[idx + global_id] = sigma[idx + global_id] * 0.5f; // b
		coeff4[idx + global_id] = (sigma[idx + 1 + global_id] - sigma[idx + global_id]) * c_1_6 * h_inv; // a
		coeff2[idx + global_id] = (y[idx + 1 + j * BLOCK_LEN].x - y[idx + j * BLOCK_LEN].x) * h_inv - h * (2.0f * sigma[idx + global_id] + sigma[idx + 1 + global_id]) * c_1_6; // c
	}
}