#include <iostream>
#include <cmath>
#include "AudioFile.h"

int main(void)
{
	//---------------------------------------------------------------
    std::cout << "**********************" << std::endl;
    std::cout << "Running Example: Load Audio File and Print Summary" << std::endl;
    std::cout << "**********************" << std::endl << std::endl;
    
    //---------------------------------------------------------------
    // 1. Set a file path to an audio file on your machine
    const std::string filePath = "../../Inspelningar/Test2/test.wav";
    
    //---------------------------------------------------------------
    // 2. Create an AudioFile object and load the audio file
    
    AudioFile<float> a;
    bool loadedOK = a.load (filePath);
    
    /** If you hit this assert then the file path above
     probably doesn't refer to a valid audio file */
    assert (loadedOK);
    
    //---------------------------------------------------------------
    // 3. Let's print out some key details
    
    std::cout << "Bit Depth: " << a.getBitDepth() << std::endl;
    std::cout << "Sample Rate: " << a.getSampleRate() << std::endl;
    std::cout << "Num Channels: " << a.getNumChannels() << std::endl;
    std::cout << "Length in Seconds: " << a.getLengthInSeconds() << std::endl;
    std::cout << "Number of Samples: " << a.getNumSamplesPerChannel() << std::endl;
    std::cout << std::endl;
}