/*
* $Id$
*
* This program uses the PortAudio Portable Audio Library.
* For more information see: http://www.portaudio.com
* Copyright (c) 1999-2000 Ross Bencina and Phil Burk
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
* ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
* CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
* The text above constitutes the entire PortAudio license; however,
* the PortAudio community also makes the following non-binding requests:
*
* Any person wishing to distribute modifications to the Software is
* requested to send the modifications to the original developer so that
* they can be incorporated into the canonical version. It is also
* requested that these non-binding requests be included along with the
* license above.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <portaudio.h>
#include <fftw3.h>

#define SAMPLE_RATE  (44100)
#define FRAMES_PER_BUFFER (512)
#define NUM_SECONDS     (5)
#define NUM_CHANNELS    (16)
#define DEVICE_NAME     "UMA16v2: USB Audio (hw:2,0)"
/* #define DITHER_FLAG     (paDitherOff) */
#define DITHER_FLAG     (0) 

#define WRITE_TO_FILE   (0)

/* Select sample format. */
#if 1
#define PA_SAMPLE_TYPE  paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE  (0.0f)
#endif

typedef struct
{
 int          frameIndex;  /* Index into sample array. */
 int          maxFrameIndex;
 SAMPLE      *recordedSamples;
}
paTestData;



/* This routine will be called by the PortAudio engine when audio is needed.
** It may be called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static int recordCallback( const void *inputBuffer, void *outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags,
                        void *userData )
{
    paTestData *data = (paTestData*)userData;
    const SAMPLE *rptr = (const SAMPLE*)inputBuffer;
    SAMPLE *wptr = &data->recordedSamples[data->frameIndex * NUM_CHANNELS];
    long framesToCalc;
    long i;
    int finished;
    unsigned long framesLeft = data->maxFrameIndex - data->frameIndex;

    (void) outputBuffer; // Prevent unused variable warnings.
    (void) timeInfo;
    (void) statusFlags;
    (void) userData;

    if( framesLeft < framesPerBuffer )
    {
        framesToCalc = framesLeft;
        finished = paComplete;
    }
    else
    {
        framesToCalc = framesPerBuffer;
        finished = paContinue;
    }

    if( inputBuffer == NULL )
    {
        for( i=0; i<framesToCalc; i++ ) 
        {            
            for (int j = 0; j < NUM_CHANNELS; j++) // each frame contains a sample point from multiple channels
            {
                *wptr++ = SAMPLE_SILENCE; // channel j
            }            
        }
    }
    else
    {
        for( i=0; i<framesToCalc; i++ )
        {            
            for (int j = 0; j < NUM_CHANNELS; j++) // each frame contains a sample point from multiple channels
            {
                *wptr++ = *rptr++; // channel j
            }            
        }
    }
    data->frameIndex += framesToCalc;
    return finished;
}

/*******************************************************************/
int main(void);
int main(void)
{
    PaStreamParameters  inputParameters;
    PaStream*           stream;
    PaError             err = paNoError;
    paTestData          data;
    int                 i;
    int                 totalFrames;
    int                 numSamples;
    int                 numBytes;    
    double              average;

    printf("patest_record.c\n"); fflush(stdout);    

    err = Pa_Initialize();
    if( err != paNoError ) {
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }

    int numDevices = Pa_GetDeviceCount();
    printf("Number of devices: %d\n", numDevices);

    if (numDevices < 0){
        printf("Error getting device count.\n");
        Pa_Terminate();        
        exit(EXIT_FAILURE);
    }
    else if (numDevices == 0){
        printf("There are no available audio devices on this machine.\n");
        Pa_Terminate();        
        exit(EXIT_FAILURE);
    }

    int device = -1;
    const PaDeviceInfo* deviceInfo;
    for (int i = 0; i < numDevices; i++)
    {
        deviceInfo = Pa_GetDeviceInfo(i);
        printf("Device %d:\n", i);
        printf("    name: %s\n", deviceInfo->name);
        printf("    maxInputChannels: %d\n", deviceInfo->maxInputChannels);
        printf("    maxOutputChannels: %d\n", deviceInfo->maxOutputChannels);
        printf("    defaultSampleRate: %f\n", deviceInfo->defaultSampleRate);

        if (strcmp(deviceInfo->name, DEVICE_NAME) == 0)
        {
            device = i;
        }
    }

    if (device == -1){
        printf("\nDevice: %s not found!\n", DEVICE_NAME);
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }

    printf("Device = %d\n", device);

    /*FILE* gnuplot = popen("gnuplot", "w");
    fprintf(gnuplot, "plot '-'\n");*/

    inputParameters.device = device;    
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.sampleFormat = PA_SAMPLE_TYPE;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    data.maxFrameIndex = totalFrames = NUM_SECONDS * SAMPLE_RATE; // Record for a few seconds.
    data.frameIndex = 0;
    numSamples = totalFrames * NUM_CHANNELS;
    numBytes = numSamples * sizeof(SAMPLE);
    data.recordedSamples = (SAMPLE *) malloc( numBytes ); // From now on, recordedSamples is initialised.
    if( data.recordedSamples == NULL )
    {
        printf("Could not allocate record array.\n");
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }
    for( i=0; i<numSamples; i++ ) data.recordedSamples[i] = 0;

    // Record some audio. --------------------------------------------
    err = Pa_OpenStream(
           &stream,
           &inputParameters,
           NULL,                  // &outputParameters
           SAMPLE_RATE,
           FRAMES_PER_BUFFER,
           paClipOff,      // we won't output out of range samples so don't bother clipping them
           recordCallback,
           &data );
    if( err != paNoError ) {
        printf("Error when opening stream.\n");
        Pa_Terminate();
        free(data.recordedSamples);
        exit(EXIT_FAILURE);
    }

    err = Pa_StartStream( stream );
    if( err != paNoError ) {
        printf("Error when starting stream.\n");
        Pa_Terminate();
        free(data.recordedSamples);
        exit(EXIT_FAILURE);
    }
    printf("\n=== Now recording!! Please speak into the microphone. ===\n"); fflush(stdout);
    //int k = 1;

    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        //Pa_Sleep(1000);
        //printf("index = %d\n", data.frameIndex ); fflush(stdout);
        //fprintf(gnuplot, "%g %g\n", k)
    }
    if( err < 0 ) {
        printf("Error when listening on stream.\n");
        Pa_Terminate();
        free(data.recordedSamples);
        exit(EXIT_FAILURE);
    }

    err = Pa_CloseStream( stream );
    if( err != paNoError ) {
        printf("Error when closing stream.\n");
        Pa_Terminate();
        free(data.recordedSamples);
        exit(EXIT_FAILURE);
    }

    // Measure maximum peak amplitude.
    float maxes[16] = {0};
    float averages[16] = {0.0f};
    float val;
    for( i=0; i<numSamples; i+= 16)
    {
        for (int j = 0; j < 16; j++)
        {
            val = data.recordedSamples[i+j];
            if( val < 0 ) val = -val; // ABS
            if( val > maxes[j] )
            {
                maxes[j] = val;
            }
            averages[j] += val;
        }
    }

    for (int i = 0; i < 16; i++)
    {
        averages[i] = averages[i] / (float)totalFrames;
        printf("Mic %d: Sample max amplitude = %.8f\n", i, maxes[i] );
        printf("Mic %d: Sample average = %lf\n", i, averages[i] );
    }

    // Write recorded data to a file.
    #if WRITE_TO_FILE
    {
        FILE  *fid;
        fid = fopen("recorded.raw", "wb");
        if( fid == NULL )
        {
            printf("Could not open file.");
        }
        else
        {
            fwrite( data.recordedSamples, NUM_CHANNELS * sizeof(SAMPLE), totalFrames, fid );
            fclose( fid );
            printf("Wrote data to 'recorded.raw'\n");
        }
    }
    #endif

    Pa_Terminate();
    free(data.recordedSamples);
    return 0;
}