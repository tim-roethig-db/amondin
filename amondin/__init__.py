"""
Module for transcribing speeches of 1 or multiple speakers
"""
from .segment_speakers import segment_speakers
from .speech2text import speech2text
from .tools import get_secret, convert_audio_to_wav
from .main import transcribe
