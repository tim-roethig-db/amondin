"""
Module containing the segment_speakers function
"""

from typing import Optional
import torch
from pyannote.audio import Pipeline, Audio


def segment_speakers(
    audio: dict,
    hf_token: str,
    device: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    tolerance: float = 0.0,
) -> list[dict]:
    """
    Detect speakers in audio.wav file and label the segments of each speaker accordingly
    :param max_speakers:
    :param min_speakers:
    :param device: Device to run the model on
    :param audio:
    :param hf_token: HF token since the pyannote model needs authentication
    :param num_speakers: Set to None to self detect the number of speakers
    :param tolerance:
    :return:
    """

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # load pipeline to device, either cuda or cpu
    pipeline.to(torch.device(device))

    # inference on the whole file
    annotation = pipeline(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # merge passages from same speaker if occurring in less than tolerance after each other
    annotation = annotation.support(tolerance)

    # get the time stamps of each speakers passage
    segments = annotation.get_timeline()

    # store all passages in a list of dicts
    speaker_segments = []
    for segment in segments:
        if segment.duration > tolerance:
            # get audio passages as numpy array
            waveform, sample_rate = Audio().crop(audio, segment, mode="pad")
            waveform = torch.squeeze(waveform)
            waveform = waveform.numpy()

            # get speaker belonging to audio
            speaker = annotation.get_labels(segment)

            # craft dict representing passage
            segment = {
                "speaker": list(speaker)[0],
                "start": segment.start,
                "end": segment.end,
                "audio": {
                    "raw": waveform,
                    "sampling_rate": sample_rate,
                },
            }

            speaker_segments.append(segment)

    return speaker_segments
