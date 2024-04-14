import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def speech2text(audio: dict, model: str = "openai/whisper-tiny", language: str = "german") -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # load model from huggingface
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=torch_dtype)
    model.to(device)

    # specify task and language
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # create input
    input_features = processor(audio["raw"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    input_features = input_features.to(torch_dtype).to(device)

    # run inference
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # convert output to text
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # return sting in list
    return result[0]
