import pandas as pd


def merge_rows_consecutive_speaker(transcript: pd.DataFrame) -> pd.DataFrame:
    transcript['speaker_group'] = (transcript['speaker'] != transcript['speaker'].shift()).cumsum()

    print(transcript.to_markdown())

    transcript = transcript.groupby(['speaker_group', 'speaker']).agg({
        'time_stamp': lambda x: ' '.join(x),
        'text': lambda x: ' '.join(x)
    }).reset_index()

    transcript = transcript.drop(columns='speaker_group')

    print(transcript.to_markdown())

    return transcript


if __name__ == "__main__":
    test_transcript = pd.read_excel("../data/test_transcript.xlsx")
    merge_rows_consecutive_speaker(transcript=test_transcript)
