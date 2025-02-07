"""
Module for transcribing speeches of 1 or multiple speakers
"""

from .segment_speakers import segment_speakers
from .speech2text import speech2text
from .tools import get_secret
from .main import transcribe
from .post_processing import merge_rows_consecutive_speaker, format_time_stamp
