"""
Module containing functions for post-processing of the transcript
"""

import pandas as pd


def _seconds_to_time_stamp(seconds: float) -> str:
    """
    Function to convert seconds to a time stamp
    :param seconds:
    :return:
    """
    minutes, seconds = divmod(seconds, 60)

    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{int(minutes):02}:{int(seconds):02}:{milliseconds:03}"


def merge_rows_consecutive_speaker(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Function to merge consecutive segments of the same speaker into one segment.
    :param transcript:
    :return:
    """
    # create a column speaker_group that signals if speakers have consecutive segments
    transcript["speaker_group"] = (transcript["speaker"] != transcript["speaker"].shift()).cumsum()
    
    # group by speaker_group and speaker to merge consecutive segments
    transcript = transcript.groupby(["speaker_group", "speaker"]).agg({
        "start": "min",
        "end": "max",
        "text": " ".join
    }).reset_index()

    # drop helper column speaker_group
    transcript = transcript.drop(columns="speaker_group")

    return transcript


def format_time_stamp(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert the start and end seconds into a time range string.
    :param transcript:
    :return:
    """
    transcript["start"] = transcript["start"].apply(_seconds_to_time_stamp)
    transcript["end"] = transcript["end"].apply(_seconds_to_time_stamp)

    transcript["time_stamp"] = transcript.apply(
        lambda row: f"{row['start']} -> {row['end']}",
        axis="columns"
    )

    return transcript[["speaker", "time_stamp", "text"]]


if __name__ == "__main__":
    test_transcript = pd.read_excel("../data/test_transcript.xlsx")
    print(format_time_stamp(transcript=test_transcript).to_markdown())
