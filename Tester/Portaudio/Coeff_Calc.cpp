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
#include <complex>
#include <intrin.h>
//#include "mex.h"


void spline_init(std::complex<double>* y, double dx, std::size_t signal_length, std::complex<double>* coeff1, std::complex<double>* coeff2, std::complex<double>* coeff3, std::complex<double>* coeff4)
{
	//Based on http://www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture11.pdf
	double h = dx;
	double h2 = h * h;
	double h_inv = 1.0 / h;
	double h2_inv = 1.0 / h2;
	const double c_1_6 = 1.0 / 6.0;
	int n = int(signal_length);
	//std::complex<double>* d = new std::complex<double>[n];
	std::complex<double>* cp = new std::complex<double>[n];
	std::complex<double>* dp = new std::complex<double>[n];
	std::complex<double>* d = new std::complex<double>[n];
	std::complex<double>* sigma = new std::complex<double>[n];
	sigma[0] = 0.0;
	sigma[n - 1] = 0.0;
	for (int idx = 1; idx < n - 1; ++idx)
	{
		d[idx] = 6.0 *h2_inv * (y[idx + 1] - 2.0 * y[idx] + y[idx - 1]);
	}

	//Using Tridiagonal matrix algorithm
	//Thomas algorithm (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)

	double a_0 = 1.0;
	double b_0 = 4.0;
	double c_0 = 1.0;
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
		coeff1[idx] = y[idx]; // d
		coeff3[idx] = sigma[idx] * 0.5; //b
		coeff4[idx] = (sigma[idx + 1] - sigma[idx]) *c_1_6*h_inv; //a
		coeff2[idx] = (y[idx + 1] - y[idx]) *h_inv - h * (2.0 * sigma[idx] + sigma[idx + 1]) *c_1_6; //c
	}

	delete[] cp,d,dp,sigma;

	
}
