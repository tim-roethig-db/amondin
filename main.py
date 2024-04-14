import pandas as pd

from diarize_speakers import diarize_speakers
from speech2text import speech2text


def main(
        file_path: str, language: str = "german", num_speakers: int = None, device: str = "cpu",
        s2t_model: str = "openai/whisper-tiny"
):
    print("Diarizing speakers...")
    diarized_speakers = diarize_speakers(file_path, num_speakers=num_speakers, device=device)

    print("Transcripting audio...")
    transcript = list()
    for i, speaker_section in enumerate(diarized_speakers):
        print(f"Transcripting part {i+1} of {len(diarized_speakers)}")
        text = speech2text(
            speaker_section["audio"],
            model=s2t_model,
            language=language,
            device=device,
        )

        transcript.append(
            [speaker_section["speaker"], speaker_section["time_stamp"], text]
        )

    # Store transcript in pandas Data Frame
    transcript = pd.DataFrame(data=transcript, columns=["speaker", "time_stamp", "text"])

    # save transcript
    print(transcript.to_markdown(index=False))


if __name__ == "__main__":
    main(
        "data/sample.wav",
        device="cpu",
        s2t_model="openai/whisper-tiny",
        language="german",
        num_speakers=2
    )
