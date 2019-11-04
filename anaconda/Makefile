#!/bin/sh

.PHONY: clean build upload

build:
	time conda build . -c pytorch --output-folder build

upload:
	anaconda upload build/noarch/yews*

clean:
	rm -rfv build/*
