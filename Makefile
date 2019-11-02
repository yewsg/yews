#!/bin/sh

.PHONY: clean test anaconda pypi build wait release

release: test wait clean build wait
	anaconda upload $(shell conda build . --output)
	twine upload dist/* -r pypi

test:
	pytest -m "not large_download" --cov-config .coveragerc --cov yews tests

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
	rm -rf build dist .eggs htmlcov *.npy *.tar.bz2 ._* .coverage .pytest_cache
