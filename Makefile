#!/bin/sh

.PHONY: clean test coverage conda pypi

conda:
	conda build . -c pytorch
	anaconda upload $(shell conda build . --output)

pypi:
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/* -r pypi

test:
	pytest -v --cov-report term-missing --cov yews tests

coverage:
	coverage html

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf dist


