VST_PATH ?= "./data/vst/MikaMicro64.dll"
MIMIC_AUDIO_PATH ?= "./data/singer.wav"

fetch-resources:
ifeq ($(OS),Windows_NT)
	@echo "Fetching resources..."
	mkdir -p data/vst/
	@curl https://freevstplugins.net/wp-content/uploads/get-it/Bitsonic_Keyzone_Classic_x64.zip -o data/vst/Bitsonic_Keyzone_Classic_x64.zip
	@unzip data/vst/Bitsonic_Keyzone_Classic_x64.zip -d data/vst/

	@echo "Install VST - Install into the ./data/vst/ folder or set the VST_PATH environment variable to point to the .dll location"
	@data/vst/Bitsonic_Keyzone_Classic_x64.exe

	rm data/vst/Bitsonic_Keyzone_Classic_x64.zip data/vst/Bitsonic_Keyzone_Classic_x64.exe

else
	@echo "Only Windows is supported currently"

endif

setup:
	mkdir -p data/midi/ data/audio/ data/fft/
	poetry run python \
		talking_piano/main.py prepare-components \
		--data-dir ./data \
		--vst-path ${VST_PATH}

mimic:
	poetry run python \
		talking_piano/main.py mimic \
		--data-dir ./data \
		--audio-path ${MIMIC_AUDIO_PATH}
