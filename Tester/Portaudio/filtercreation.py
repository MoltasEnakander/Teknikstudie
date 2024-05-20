import scipy
import numpy as np

def filtercreation(nFilters, nTaps, bandwidth):
	assert(nFilters > 0)	

	coeffs = np.asarray(scipy.signal.firwin(nTaps, bandwidth)) # first filter is a lowpass filter	

	for i in range(1, nFilters):
		# add bandpass filters		
		c = scipy.signal.firwin(nTaps, np.array([bandwidth*i, bandwidth*(i+1)]))		
		coeffs = np.append(coeffs, c) # remaining filters are bandpass filters
	
	return coeffs