"""
Module containing the speech to text function
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def speech2text(
        audio: dict,
        device: str,
        model: str,
        language: str
) -> str:
    """
    Translate audio to text
    :param device: Device to run the model on [cpu, cuda or cuda:x]
    :param audio: dictionary containing audio as numpy array of shape (n,) and the sampling rate
    :param model:
    :param language:
    :return:
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # load model from huggingface
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch_dtype
    ).to(device)

    # specify task and language
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )

    # create input
    input_features = processor(
        audio["raw"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features.to(torch_dtype).to(device)

    # run inference
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids
    )

    # convert output to text
    result = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )

    # return sting in list
    return result[0]
