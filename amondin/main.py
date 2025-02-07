"""
Main module of transcription tool
"""

import pandas as pd
import torchaudio

from amondin.tools import get_secret
from amondin.segment_speakers import segment_speakers
from amondin.speech2text import speech2text
from amondin.post_processing import merge_rows_consecutive_speaker, format_time_stamp


def transcribe(
    input_file_path: str,
    output_file_path: str,
    hf_token: str,
    device: str = "cpu",
    language: str = None,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    s2t_model: str = "openai/whisper-tiny",
    tolerance: float = 0.0,
):
    """
    Transcribe a give audio.wav file.
    :param max_speakers:
    :param min_speakers:
    :param tolerance: Seconds of silence between the same speaker to still merge the segments
    :param device: Device to run the model on [cpu, cuda or cuda:x]
    :param output_file_path:
    :param input_file_path:
    :param hf_token: You Huggingface token to access gated models
    :param language: Set the language for improved performance. None results in language detection.
    :param num_speakers: Set the number of speakers for improved performance. None results in
    auto-detection.
    :param s2t_model: Select the speach to text model you want to use (tested on openai/whisper-*)
    :return:
    """

    print(f"Running on {device}...")

    print(f"Loading {input_file_path}...")
    waveform, sample_rate = torchaudio.load(input_file_path)
    audio = {"waveform": waveform, "sample_rate": sample_rate}

    print("Segmenting speakers...")
    segments = segment_speakers(
        audio,
        hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
        tolerance=tolerance,
    )

    print("Transcribing audio...")
    transcript = speech2text(
        [segment["audio"] for segment in segments],
        model_name=s2t_model,
        language=language,
        device=device,
    )

    for i, segment in enumerate(segments):
        del segment["audio"]
        segment["text"] = transcript[i]

    transcript = pd.DataFrame(segments)

    print("Post processing...")
    transcript = merge_rows_consecutive_speaker(transcript)

    transcript = format_time_stamp(transcript)

    print("Saving transcript...")
    print(transcript.to_markdown(index=False))
    if output_file_path.endswith(".csv"):
        transcript.to_csv(output_file_path, index=False, sep=";")
    elif output_file_path.endswith(".xlsx"):
        transcript.to_excel(output_file_path, index=False)
    else:
        raise TypeError("Only .csv and .xlsx are valid file types.")


if __name__ == "__main__":
    transcribe(
        "../data/sample.wav",
        "../data/sample.xlsx",
        hf_token=get_secret("../secrets.yaml", "hf-token"),
        s2t_model="openai/whisper-tiny",
        device="cpu",
        language="german",
        num_speakers=2,
    )
