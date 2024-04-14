from transformers import WhisperProcessor, WhisperForConditionalGeneration


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
    input_features.to(device)

    # run inference
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # convert output to text
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # return sting in list
    return result[0]
