from amondin import get_secret, diarize_speakers, speech2text, transcribe, convert_audio_to_wav


# print(diarize_speakers("./data/sample.wav", hf_token=get_secret("./secrets.yaml", "hf-token")))

# diarized_speakers = diarize_speakers("./data/sample.wav", hf_token=get_secret("./secrets.yaml", "hf-token"))
# print(diarized_speakers)
# print(speech2text(diarized_speakers[0]["audio"]))

transcribe(
    "./data/sample_short.wav",
    hf_token=get_secret("./secrets.yaml", "hf-token"),
    s2t_model="openai/whisper-tiny",
    language=None,
    num_speakers=2
)

# convert_audio_to_wav("data/sample_short.mp3", "data/sample_short.wav")
