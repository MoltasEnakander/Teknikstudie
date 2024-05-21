#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../CPU/beamformer_wideband.h"
//#include "gpu_beamformer.h"
#include "AudioFile.h"

#include <dirent.h>
#include <string>
#include <vector>

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

		struct dirent *entry = nullptr;
		DIR* dp = nullptr;

		dp = opendir(filepath);
		
		std::vector<AudioFile<float>> files(16);
		AudioFile<float> a;

		bool loadedOK;

		if (dp != nullptr)
		{
			while((entry = readdir(dp))){
				std::string s = entry->d_name;
				if (s.find(".wav") != std::string::npos){					
					char result[100];
					strcpy(result, filepath);
					strcat(result, entry->d_name);
					loadedOK = a.load(result);

					assert(loadedOK);

					if (s.find("11.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[10] = a;
					}
					else if (s.find("12.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[11] = a;
					}
					else if (s.find("13.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[12] = a;
					}
					else if (s.find("14.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[13] = a;
					}
					else if (s.find("15.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[14] = a;
					}
					else if (s.find("16.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[15] = a;
					}
					else if (s.find("1.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[0] = a;
					}
					else if (s.find("2.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[1] = a;
					}
					else if (s.find("3.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[2] = a;
					}
					else if (s.find("4.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[3] = a;
					}
					else if (s.find("5.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[4] = a;
					}
					else if (s.find("6.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[5] = a;
					}
					else if (s.find("7.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[6] = a;
					}
					else if (s.find("8.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[7] = a;
					}
					else if (s.find("9.wav") != std::string::npos){
						printf("%s\n", entry->d_name);
						files[8] = a;
					}
				}					
			}				
		}

		/*for (int i = 0; i < NUM_CHANNELS; ++i)
		{
			// make sure that all channels have the same amount of samples
			assert(files[i].getNumSamplesPerChannel() > 0);
			assert(files[0].getNumSamplesPerChannel() == files[i].getNumSamplesPerChannel());
		}*/

		closedir(dp);

		//listen_prerecorded(files);

		/*for (int i = 0; i < 16; ++i)
		{
			printf("%d: \n", i);
			std::cout << "Bit Depth: " << files[i].getBitDepth() << std::endl;
	    	std::cout << "Sample Rate: " << files[i].getSampleRate() << std::endl;	    
	    	std::cout << "Length in Seconds: " << files[i].getLengthInSeconds() << std::endl;
	    	std::cout << "Number of Samples: " << files[i].getNumSamplesPerChannel() << std::endl;
	    	std::cout << std::endl;
		}*/
	}
	else // listen live to UMA16 device
	{
		printf("Using live recording!!\n");
		listen_live();
	}
	return 0;
}