// 2020-03-30
// Cubic spline Interpolation of complex signals
// Input parameters
// complex signals (sampled at fixed frequency)
// sampel frequency/time values for complex values (knots)
// interpolation points
// Output parameters
// interpolated values at interpolation points
//
// Description:
// Intilization: Calculate coefficients used for cubic spline interpolation
// Sets up 4x arrays for the cubic spline interpolation based on knots
// Processing: For each interpolation point calculate interpolation values
//#include <complex>
//#include <intrin.h>
//#include "mex.h"
#include "Coeff_Calc.h"

void spline_init(fftw_complex* y, float dx, std::size_t signal_length, float* coeff1, float* coeff2, float* coeff3, float* coeff4)
{
	//Based on http://www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture11.pdf
	float h = dx;
	float h2 = h * h;
	float h_inv = 1.0f / h;
	float h2_inv = 1.0f / h2;
	const float c_1_6 = 1.0f / 6.0f;
	int n = int(signal_length);
	
	float* cp = new float[n];
	float* dp = new float[n];
	float* d = new float[n];
	float* sigma = new float[n];
	sigma[0] = 0.0f;
	sigma[n - 1] = 0.0f;
	for (int idx = 1; idx < n - 1; ++idx)
	{
		d[idx] = 6.0f * h2_inv * (y[idx + 1][0] - 2.0f * y[idx][0] + y[idx - 1][0]);
	}

	//Using Tridiagonal matrix algorithm
	//Thomas algorithm (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)

	float a_0 = 1.0f;
	float b_0 = 4.0f;
	float c_0 = 1.0f;
	cp[1] = c_0 / b_0;
	dp[1] = d[1]/b_0;


	for (int idx = 2; idx < n - 1; ++idx)
	{
		cp[idx] = c_0 / (b_0 - a_0 * cp[idx - 1]);
		dp[idx] = (d[idx] - a_0 * dp[idx - 1]) / (b_0 - a_0 * cp[idx - 1]);
	}
	sigma[n - 2] = dp[n - 2];

	for (int idx = n - 3; idx > 0; --idx)
	{
		sigma[idx] = dp[idx] - cp[idx] * sigma[idx + 1];
	}
	for (int idx = 0; idx < n - 1; ++idx)
	{
		coeff1[idx] = y[idx][0]; // d
		coeff3[idx] = sigma[idx] * 0.5f; //b
		coeff4[idx] = (sigma[idx + 1] - sigma[idx]) * c_1_6 * h_inv; //a
		coeff2[idx] = (y[idx + 1][0] - y[idx][0]) * h_inv - h * (2.0f * sigma[idx] + sigma[idx + 1]) * c_1_6; //c
	}

	delete[] cp,d,dp,sigma;
}


