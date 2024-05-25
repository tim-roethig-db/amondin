from datetime import timedelta
import pandas as pd


def _seconds_to_time_stamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)

    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{int(minutes):02}:{int(seconds):02}:{milliseconds:03}"


def merge_rows_consecutive_speaker(transcript: pd.DataFrame) -> pd.DataFrame:
    transcript['speaker_group'] = (transcript['speaker'] != transcript['speaker'].shift()).cumsum()

    transcript = transcript.groupby(['speaker_group', 'speaker']).agg({
        'start': "min",
        "end": "max",
        'text': lambda x: ' '.join(x)
    }).reset_index()

    transcript = transcript.drop(columns='speaker_group')

    return transcript


def format_time_stamp(transcript: pd.DataFrame) -> pd.DataFrame:
    transcript['start'] = transcript['start'].apply(_seconds_to_time_stamp)
    transcript['end'] = transcript['end'].apply(_seconds_to_time_stamp)

    transcript['time_stamp'] = transcript.apply(
        lambda row: f"{row['start']} -> {row['end']}",
        axis='columns'
    )

    return transcript[['speaker', "time_stamp", "text"]]


if __name__ == "__main__":
    test_transcript = pd.read_excel("../data/test_transcript.xlsx")
    print(format_time_stamp(transcript=test_transcript).to_markdown())
