# Amondin
A simple transcription tool able to segment speakers and convert audio to text.

## Features

- convert audio to text
- segment speakers
- supporting multiple file formats (.wav is guaranteed to work)
- detecting number of speakers (hard setting improves performance)
- detecting language (hard setting improves performance)
- return transcribed dialogs as .csv

## Installation
    pip install git+https://github.com/tim-roethig-db/transcription-tool.git@master

## Usage
    from amondin import transcribe

    transcribe(
        "sample.m4a", "sample.csv",
        hf_token="<your hf token>",
        s2t_model="openai/whisper-large-v3",
        device="cuda",
        language="german",
        num_speakers=2
    )