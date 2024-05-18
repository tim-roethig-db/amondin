"""
Main module of transcription tool
"""

import pandas as pd
from pathlib import Path

from amondin.tools import convert_audio_to_wav
from amondin.diarize_speakers import diarize_speakers
from amondin.speech2text import speech2text


def transcribe(
        input_file_path: str, output_file_path: str, hf_token: str, device: str = "cpu",
        language: str = "german", num_speakers: int = None, s2t_model: str = "openai/whisper-tiny"
):
    """
    Transcribe a give audio.wav file.
    :param device: Device to run the model on [cpu, cuda or cuda:x]
    :param output_file_path:
    :param input_file_path:
    :param hf_token:
    :param language: Set the language for improved performance. None results in language detection.
    :param num_speakers: Set the number of speakers for improved performance. None results in
    auto-detection.
    :param s2t_model:
    :return:
    """

    print(f"Running on {device}...")

    if not input_file_path.endswith(".wav"):
        print("Converting audio to .wav...")
        file_name = Path(input_file_path).stem
        print(file_name)
        convert_audio_to_wav(input_file_path, f"{file_name}.wav")
        input_file_path = f"{file_name}.wav"

    print("Diarizing speakers...")
    diarized_speakers = diarize_speakers(
        input_file_path,
        hf_token=hf_token,
        num_speakers=num_speakers,
        device=device
    )

    print("Transcribing audio...")
    transcript = []
    for i, speaker_section in enumerate(diarized_speakers):
        print(f"Transcribing part {i+1} of {len(diarized_speakers)}")
        text = speech2text(
            speaker_section["audio"],
            model=s2t_model,
            language=language,
            device=device
        )

        transcript.append(
            [speaker_section["speaker"], speaker_section["time_stamp"], text]
        )

    # Store transcript in pandas Data Frame
    transcript = pd.DataFrame(data=transcript, columns=["speaker", "time_stamp", "text"])

    # save transcript
    print(transcript.to_markdown(index=False))
    transcript.to_csv(output_file_path, index=False, sep=";")
