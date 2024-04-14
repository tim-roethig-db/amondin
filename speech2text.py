from transformers import WhisperProcessor, WhisperForConditionalGeneration

from diarize_speakers import diarize_speakers


def speech2text(audio: dict, model: str = "openai/whisper-tiny", device: str = "cpu", language: str = "german") -> str:
    # load model from huggingface
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(model)

    # load pipeline to device, either cuda or cpu
    model.to(device)

    # specify task and language
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # create input
    input_features = processor(audio["raw"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    # run inference
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # convert output to text
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # return sting in list
    return result[0]


if __name__ == "__main__":
    diarized_speakers = diarize_speakers("data/sample.wav")
    print(diarized_speakers)
    print(speech2text(diarized_speakers[0]["audio"]))
