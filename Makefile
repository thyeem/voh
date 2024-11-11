.PHONY: build
build:
	pip install -U pip
	pip install Cython
	pip install pybind11
	pip install wheel setuptools pip --upgrade
	pip install fasttext
	pip install youtokentome
	pip install -U foc ouch
	pip install -U pytubefix
	pip install -r requirements.txt
	@which ffmpeg > /dev/null 2>&1 || brew install ffmpeg
	@which sox > /dev/null 2>&1 || brew install sox
