EXEC = cppaudiocapture

CLIB = -I./lib/portaudio/include ./lib/portaudio/lib/.libs/libportaudio.a -lrt -lasound -ljack -pthread

$(EXEC): main.cpp
	g++ -o build/$@ $^ $(CLIB)

install-deps:
	mkdir -p lib

	wget http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz && tar -xvf pa_stable_v190700_20210406.tgz -C lib && rm pa_stable_v190700_20210406.tgz
	cd lib/portaudio && ./configure && $(MAKE) -j
.PHONY: install-deps

uninstall-deps:
	cd lib/portaudio && $(MAKE) uninstall
	rm -rf lib/portaudio
.PHONY: uninstall-deps

clean:
	rm -f $(EXEC)
.PHONY: clean

