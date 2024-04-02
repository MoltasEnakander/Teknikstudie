#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string.h>
#include <portaudio.h>

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 512

static void checkErr(PaError err)
{
	if (err != paNoError)
	{
		printf("PortAudio error: %s\n", Pa_GetErrorText(err));
		exit(1);
	}
}

static inline float max(float a, float b)
{
	return a > b ? a : b;
}

/*static inline float abs(float a)
{
	return a > 0 ? a : -a;
}*/

static int patestCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
							const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
							void* userData)
{
	float* in = (float*)inputBuffer;
	(void)outputBuffer;

	int dispSize = 100;
	printf("\r");

	float vol_1 = 0;
	float vol_2 = 0;
	float vol_3 = 0;
	float vol_4 = 0;
	float vol_5 = 0;
	float vol_6 = 0;
	float vol_7 = 0;
	float vol_8 = 0;
	float vol_9 = 0;
	float vol_10 = 0;
	float vol_11 = 0;
	float vol_12 = 0;
	float vol_13 = 0;
	float vol_14 = 0;
	float vol_15 = 0;
	float vol_16 = 0;

	for (unsigned long i = 0; i < framesPerBuffer * 16; i += 16)
	{		
		vol_1 = max(vol_1, std::abs(in[i]));
		vol_2 = max(vol_2, std::abs(in[i+1]));
		vol_3 = max(vol_3, std::abs(in[i+2]));
		vol_4 = max(vol_4, std::abs(in[i+3]));
		vol_5 = max(vol_5, std::abs(in[i+4]));
		vol_6 = max(vol_6, std::abs(in[i+5]));
		vol_7 = max(vol_7, std::abs(in[i+6]));
		vol_8 = max(vol_8, std::abs(in[i+7]));
		vol_9 = max(vol_9, std::abs(in[i+8]));
		vol_10 = max(vol_10, std::abs(in[i+9]));
		vol_11 = max(vol_11, std::abs(in[i+10]));
		vol_12 = max(vol_12, std::abs(in[i+11]));
		vol_13 = max(vol_13, std::abs(in[i+12]));
		vol_14 = max(vol_14, std::abs(in[i+13]));
		vol_15 = max(vol_15, std::abs(in[i+14]));
		vol_16 = max(vol_16, std::abs(in[i+15]));
	}

	float vols[16] = {vol_1, vol_2, vol_3, vol_4, vol_5, vol_6, vol_7, vol_8, vol_9, vol_10, vol_11, vol_12, vol_13, vol_14, vol_15, vol_16};

	/*for (int j = 0; j < 16; j++)
	{*/
		for (int i = 0; i < dispSize; i++)
		{
			float barProportion = i / (float)dispSize;
			if (vols[0] >= barProportion){
				printf("█");
			}
			else{
				printf(" ");
			}
		}
		//printf("\n");
	//}
	

	fflush(stdout);

	return 0;
}


int main() 
{
	PaError err = Pa_Initialize();
	checkErr(err);

	int numDevices = Pa_GetDeviceCount();
	printf("Number of devices: %d\n", numDevices);

	if (numDevices < 0){
		printf("Error getting device count.\n");
		exit(1);
	}
	else if (numDevices == 0){
		printf("There are no available audio devices on this machine.\n");
		exit(1);
	}

	int device = -1;
	const char *str = "UMA16v2: USB Audio (hw:2,0)";

	const PaDeviceInfo* deviceInfo;
	for (int i = 0; i < numDevices; i++)
	{
		deviceInfo = Pa_GetDeviceInfo(i);
		printf("Device %d:\n", i);
		printf("	name: %s\n", deviceInfo->name);
		printf("	maxInputChannels: %d\n", deviceInfo->maxInputChannels);
		printf("	maxOutputChannels: %d\n", deviceInfo->maxOutputChannels);
		printf("	defaultSampleRate: %f\n", deviceInfo->defaultSampleRate);

		if (strcmp(deviceInfo->name, str) == 0)
		{
			device = i;
		}
	}

	if (device == -1){
		printf("UMA16v2: USB Audio (hw:2,0) not found.\n");
		exit(1);
	}

	printf("Device = %d\n", device);

	PaStreamParameters inputParameters;
	PaStreamParameters outputParameters;

	memset(&inputParameters, 0, sizeof(inputParameters));
	inputParameters.channelCount = Pa_GetDeviceInfo(device)->maxInputChannels;
	inputParameters.device = device;
	inputParameters.hostApiSpecificStreamInfo = NULL;
	inputParameters.sampleFormat = paFloat32;
	inputParameters.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

	memset(&outputParameters, 0, sizeof(outputParameters));
	outputParameters.channelCount = Pa_GetDeviceInfo(device)->maxOutputChannels;
	outputParameters.device = device;
	outputParameters.hostApiSpecificStreamInfo = NULL;
	outputParameters.sampleFormat = paFloat32;
	outputParameters.suggestedLatency = Pa_GetDeviceInfo(device)->defaultLowInputLatency;

	PaStream* stream;
	err = Pa_OpenStream(
		&stream,
		&inputParameters,
		&outputParameters,
		SAMPLE_RATE,
		FRAMES_PER_BUFFER,
		paNoFlag,
		patestCallback,
		NULL
	);
	checkErr(err);

	err = Pa_StartStream(stream);
	checkErr(err);

	Pa_Sleep(10 * 1000);

	err = Pa_StopStream(stream);
	checkErr(err);

	err = Pa_CloseStream(stream);
	checkErr(err);

	err = Pa_Terminate();
	checkErr(err);

	return 0;
}
