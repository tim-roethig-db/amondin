"""
Module containing the speech to text function
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def speech2text(
        audio: dict,
        device: str,
        model: str,
        language: str
) -> str:
    """
    Translate audio to text
    :param device: Device to run the model on [cpu, cuda or cuda:x]
    :param audio: dictionary containing audio as numpy array of shape (n) and the sampling rate
    :param model:
    :param language:
    :return:
    """
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    """
    # load model from huggingface
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch_dtype,
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
        return_tensors="pt",
    ).input_features.to(torch_dtype).to(device)

    # run inference
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
    )

    # convert output to text
    result = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )
    """

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio)
    print(result)
    return result["text"]

    # return sting in list
    return result[0]
