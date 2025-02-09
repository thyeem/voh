.PHONY: build
build:
	@pip install -U \
		pip \
		foc \
		ouch \
		pytubefix
	@pip install \
		ipython \
		torch \
		"numpy<2" \
		librosa \
		soundfile \
		sounddevice \
		click

	@if [ "$(shell uname -s)" = "Linux" ]; then \
		sudo apt-get install -y ffmpeg sox portaudio19-dev; \
	else \
		which ffmpeg > /dev/null 2>&1 || brew install ffmpeg; \
		which sox > /dev/null 2>&1 || brew install sox; \
	fi

.PHONY: nemo
nemo:
	@pip install \
		Cython \
		pybind11 \
		wheel \
		setuptools \
		fasttext \
		youtokentome \
		huggingface-hub==0.23.2 \
		nemo_toolkit[asr] \
		nemo_toolkit[nlp] \
		hydra-core \
		pytorch_lightning \
		transformers
