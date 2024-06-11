#include <complex>
#include <intrin.h>
//Cubic spline interpolation, using coefficients precalculated in interp_init. långsammare än matlab...
void splinter(double dt, std::size_t signal_length, std::complex<double>* coeff1, std::complex<double>* coeff2, std::complex<double>* coeff3, std::complex<double>* coeff4, double* values, std::size_t values_length, std::complex<double>* output)
{
	std::size_t knot;
	double mu;
	double mu2, mu3;
	
	
	double dt_inv = 1 / dt;
	
	
	

	
	for (std::size_t idx = 0; idx < values_length; ++idx)
	{
		mu = (values[idx]);
		knot = (std::size_t)(mu * dt_inv + 1.0e-15);//(std::size_t)floor(mu * dt_inv);
		mu = mu - (double)knot*dt;
		//mexPrintf("idx: %i value: %5.5f knot: %i mu: %5.5f\n", idx, values[idx], knot, mu);
		mu2 = mu * mu;
		mu3 = mu2 * mu;
		//mexPrintf("Real Coefficients: %5.5f %5.5f %5.5f %5.5f\n", coeff1[knot].real(), coeff2[knot].real(), coeff3[knot].real(), coeff4[knot].real());
		output[idx] = coeff1[knot] + coeff2[knot] * mu + coeff3[knot] * mu2 + coeff4[knot] * mu3;
		//	}
	}

	
}

