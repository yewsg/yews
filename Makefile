#!/bin/sh

.PHONY: clean test anaconda pypi build wait release

release: test wait build wait
	anaconda upload $(shell conda build . --output)
	twine upload dist/* -r pypi

test:
	python setup.py test
	coverage report

wait:
	sleep 10

build: anaconda pypi

anaconda:
	time conda build . -c pytorch

pypi:
	python setup.py dists
	twine check dist/*

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf build dist yews.egg-info htmlcov *.npy *.tar.bz2
