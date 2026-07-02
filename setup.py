from setuptools import setup, find_packages

setup(
    name="dissonance",
    version="0.1.0",
    description="Unpleasant sound generator and optimizer",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "soundfile",
        "librosa",
    ],
    entry_points={
        "console_scripts": [
            "dissonance=dissonance.cli:main",
        ],
    },
)
