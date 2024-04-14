from transcription_tool import get_secret, diarize_speakers, speech2text, transcribe


# print(diarize_speakers("./data/sample.wav", hf_token=get_secret("./secrets.yaml", "hf-token")))

# diarized_speakers = diarize_speakers("./data/sample.wav", hf_token=get_secret("./secrets.yaml", "hf-token"))
# print(diarized_speakers)
# print(speech2text(diarized_speakers[0]["audio"]))

transcribe(
    "./data/sample.wav",
    hf_token=get_secret("./secrets.yaml", "hf-token"),
    s2t_model="openai/whisper-tiny",
    language="german",
    num_speakers=2
)