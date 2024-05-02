#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu_beamformer.h"
#include "AudioFile.h"

int main(int argc, char** argv)
{	
	int c;
	bool useRecording = false;
	char* filepath = NULL;
	while((c = getopt(argc, argv, "p:")) != -1)
	{
		switch(c){
			case 'p':
				useRecording = true;
				filepath = optarg;
				
				break;

			default:
				break;
		}
	}

	if (useRecording) // use recorded signal
	{
		printf("Using previous recording!\n");
		printf("%s\n", filepath);

		AudioFile* files[16];

		

		AudioFile<float> a;
    	bool loadedOK = a.load (filepath);

    	assert (loadedOK);
    	std::cout << "Bit Depth: " << a.getBitDepth() << std::endl;
	    std::cout << "Sample Rate: " << a.getSampleRate() << std::endl;	    
	    std::cout << "Length in Seconds: " << a.getLengthInSeconds() << std::endl;
	    std::cout << "Number of Samples: " << a.getNumSamplesPerChannel() << std::endl;
	    std::cout << std::endl;
	}
	else // listen live to UMA16 device
	{
		printf("Using live recording!!\n");
		listen_live();
	}
	return 0;
}