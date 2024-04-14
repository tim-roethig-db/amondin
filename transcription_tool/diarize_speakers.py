import torch
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment


def diarize_speakers(
        file_path: str, hf_token: str, num_speakers: int = None, tolerance: float = 1.0
) -> list[dict]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # load pipeline to device, either cuda or cpu
    pipeline.to(torch.device(device))

    # inference on the whole file
    #annotation = pipeline(file_path, num_speakers=num_speakers)

    # inference on excerpt
    excerpt = Segment(start=0.0, end=30.0)
    waveform, sample_rate = Audio().crop(file_path, excerpt)
    annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)

    # merge passages from same speaker if occurring in less then tolerance after each other
    annotation = annotation.support(tolerance)

    # get the time stamps of each speakers passage
    segments = annotation.get_timeline()

    # store all passages in a list of dicts
    diarized_speakers = list()
    for segment in segments:
        # get audio passages as numpy array
        waveform, sample_rate = Audio().crop(file_path, segment)
        waveform = torch.squeeze(waveform)
        waveform = waveform.numpy()

        # get speaker belonging to audio
        speaker = annotation.get_labels(segment)

        # craft dict representing passage
        segment = {
            "speaker": list(speaker)[0],
            "time_stamp": str(segment),
            "audio": {"raw": waveform, "sampling_rate": sample_rate},
        }

        diarized_speakers.append(segment)

    return diarized_speakers
