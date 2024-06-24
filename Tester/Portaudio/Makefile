EXEC6 = gpu_beamformer

EXEC13 = beamformer

CLIB = -I./lib/portaudio/include ./lib/portaudio/lib/.libs/libportaudio.a \
		-lrt -lasound -ljack -pthread -I./lib/matplot/include -I /usr/include/python3.8/ -lpython3.8 #-I./lib/fftw-3.3.10/api -lfftw3 \
		-lsfml-graphics -lsfml-window -lsfml-system	-I /usr/include/python3.8/ -lpython3.8

$(EXEC6): GPU/gpu_beamformer.cu
	nvcc -gencode arch=compute_62,code=sm_62 -rdc=true -o build/$@ $^ -I./lib/portaudio/include ./lib/portaudio/lib/.libs/libportaudio.a -lrt -lasound -ljack \
			-I./lib/matplot/include -I /usr/include/python3.8/ -lpython3.8 -lcudadevrt

$(EXEC13): beamformer/beamformer.cu beamformer/Coeff_Calc.cu beamformer/beamforming.cu beamformer/bandpass_beams.cu beamformer/Interpolation_Preparation.cpp
	nvcc -gencode arch=compute_62,code=sm_62 -rdc=true $^ -o build/$@ -I./lib/portaudio/include ./lib/portaudio/lib/.libs/libportaudio.a -lrt -lasound -ljack \
			-I./lib/matplot/include -I /usr/include/python3.8/ -lpython3.8 -lcudadevrt -I./lib/fftw-3.3.10/api -lfftw3f -lm \
			-I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft -I./extern/pybind11

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

