import yaml
import ffmpeg
import librosa
import soundfile


def get_secret(key: str):
    with open("secrets.yaml", "r") as file:
        secrets = yaml.safe_load(file)

    return secrets[key]


def convert_audio_to_wav(input_path: str, output_path: str):
    ffmpeg.input(input_path).output(
        output_path,
        format="wav",
    ).run(
        overwrite_output=True
    )

    y, s = librosa.load(output_path, sr=16000)
    soundfile.write(output_path, y, s)


if __name__ == "__main__":
    convert_audio_to_wav("data/sample.mp3", "data/sample.wav")
