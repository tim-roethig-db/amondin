"""
Module containing useful tools for amondin
"""

import yaml
import ffmpeg
import librosa
import soundfile


def get_secret(path2yaml: str, key: str):
    """
    Function to retrieve a value from a secrets.yaml
    :param path2yaml:
    :param key:
    :return:
    """
    with open(path2yaml, "r", encoding="utf-8") as file:
        secrets = yaml.safe_load(file)

    return secrets[key]


def convert_audio_to_wav(input_path: str, output_path: str):
    """
    Convert a given input audio file to .wav needed for AI pipelines
    :param input_path:
    :param output_path:
    :return:
    """
    ffmpeg.input(input_path).output(
        output_path,
        format="wav",
    ).run(
        overwrite_output=True
    )

    y, s = librosa.load(output_path, sr=16000)
    soundfile.write(output_path, y, s)
