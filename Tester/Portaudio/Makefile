EXEC = paex_record

EXEC2 = tutorial_cpp_audio_capture

EXEC3 = beamformer

EXEC4 = plot_signal

CLIB = -I./lib/portaudio/include ./lib/portaudio/lib/.libs/libportaudio.a \
		-lrt -lasound -ljack -pthread -I./lib/fftw-3.3.10/api -lfftw3		

$(EXEC): paex_record.c
	g++ -o build/$@ $^ $(CLIB)

$(EXEC2): tutorial_cpp_audio_capture.cpp
	g++ -o build/$@ $^ $(CLIB)

$(EXEC3): beamformer.cpp
	g++ -o build/$@ $^ $(CLIB)

$(EXEC4): plot_signal.cpp
	g++ -o build/$@ $^ $(CLIB)

install-deps: install-portaudio install-fftw
.PHONY: install-deps

uninstall-deps: uninstall-portaudio uninstall-fftw
.PHONY: uninstall-deps

install-portaudio:
	mkdir -p lib

	wget https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz && tar -xvf pa_stable_v190700_20210406.tgz -C lib && rm pa_stable_v190700_20210406.tgz
	cd lib/portaudio && ./configure && $(MAKE) -j
.PHONY: install-portaudio

uninstall-portaudio:
	cd lib/portaudio && $(MAKE) uninstall
	rm -rf lib/portaudio
.PHONY: uninstall-portaudio

install-fftw:
	mkdir -p lib

	wget http://fftw.org/fftw-3.3.10.tar.gz && tar -xvf fftw-3.3.10.tar.gz -C lib && rm fftw-3.3.10.tar.gz
	cd lib/fftw-3.3.10 && ./configure && $(MAKE) -j && sudo $(MAKE) install
.PHONY: install-fftw

uninstall-fftw:
	cd lib/fftw-3.3.10 && $(MAKE) uninstall
	rm -rf lib/fftw-3.3.10
.PHONY: uninstall-fftw

clean:
	rm -f $(EXEC)
.PHONY: clean

