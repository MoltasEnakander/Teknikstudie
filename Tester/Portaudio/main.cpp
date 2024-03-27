#include <stdlib.h>
#include <stdio.h>

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

	int device = 0;

	const PaDeviceInfo* deviceInfo;
	for (int i = 0; i < numDevices; i++)
	{
		deviceInfo = Pa_GetDeviceInfo(i);
		printf("Device %d:\n", i);
		printf("	name: %s\n", deviceInfo->name);
		printf("	maxInputChannels: %d\n", deviceInfo->maxInputChannels);
		printf("	maxOutputChannels: %d\n", deviceInfo->maxOutputChannels);
		printf("	defaultSampleRate: %f\n", deviceInfo->defaultSampleRate);

		if (deviceInfo->name == "UMA16v2: USB Audio (hw:2,0)")
		{
			device = i;
		}
	}

	printf("Device = %d\n", device);

	/*PaStreamParameters inputParameters;
	PaStreamParameters outputParameters;

i	memset(&inputParameters, 0, sizeof(inputParameters));
	inputParameters.channelCount = Pa_GetDeviceInfo(device)->maxOutputChannels;
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

	);*/

	err = Pa_Terminate();
	checkErr(err);

	return 0;
}
