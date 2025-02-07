"""
Module containing the speech to text function
"""

from typing import Optional
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def speech2text(
    audio: list[dict], device: str, model_name: str, language: Optional[str] = None
) -> list[str]:
    """
    Translate audio to text
    :param device: Device to run the model on [cpu, cuda or cuda:x]
    :param audio: dictionary containing audio as numpy array of shape (n) and the sampling rate
    :param model_name:
    :param language:
    :return:
    """

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    results = pipe(audio, generate_kwargs={"task": "transcribe", "language": language})

    # return string in a list
    return [result["text"] for result in results]
