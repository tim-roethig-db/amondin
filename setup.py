from setuptools import setup, find_packages


setup(
    name="transcription_tool",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "transformers==4.39.3",
        "pyannote.audio==3.1.1",
        "pyannote.core==5.0.0",
        "pyyaml==6.0.1",
        "ffmpeg-python==0.2.0",
        "pandas==2.2.2",
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "numpy==1.26.4",
        "torch==2.2.2",
    ]
)