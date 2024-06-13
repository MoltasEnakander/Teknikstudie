#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define SAMPLE_RATE (44100.0)       // How many audio samples to capture every second (44100 Hz is standard)
#define NUM_CHANNELS (16)           // Number of audio channels to capture
#define NUM_SECONDS (10)
#define DEVICE_NAME "UMA16v2: USB Audio (hw:2,0)"

#define MIN_VIEW (-60.0)
#define MAX_VIEW (60.0)
#define VIEW_INTERVAL (10.0)
#define NUM_BEAMS (int)((MAX_VIEW - MIN_VIEW) / VIEW_INTERVAL + 1)

#define MAX_THREADS_PER_BLOCK (1024)

#define NUM_TAPS (49)
#define NUM_FILTERS (6)
#define F_C (500)
#define BANDWIDTH (2 * F_C * 2 / SAMPLE_RATE)
#define BLOCK_LEN (2048)
#define TEMP (128)

#define FFT_OUTPUT_SIZE (BLOCK_LEN)

#define C (340.0) // m/s
#define ARRAY_DIST (0.042) // m

#endif