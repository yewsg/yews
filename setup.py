from setuptools import setup, find_packages

about = {}
with open("yews/__about__.py") as fp:
    exec(fp.read(), about)

requirements = [
    'numpy',
    'torch',
]

if __name__ == '__main__':

    setup(
        # metadata
        name=about['__title__'],
        version=about['__version__'],
        author=about['__author__'],
        author_email=about['__email__'],
        description="Deep learning toolbox for seismic waveform processing.",
        license=about['__license__'],

        # package info
        packages=find_packages(exclude=('test', 'examples')),

        zip_safe=True,
        install_requires=requirements,
        extras_require={
            "scipy": ["scipy"],
            "obspy": ["obspy"],
        },
    )
