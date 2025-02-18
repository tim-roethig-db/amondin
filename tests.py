"""
Tests for amondin package.
"""

from hypothesis import given, strategies as st
from amondin.post_processing import _seconds_to_time_stamp


@given(st.floats(allow_infinity=False, allow_nan=False))
def test_seconds_to_time_stamp(seconds):
    """
    Test function for _seconds_to_time_stamp from amondin.post_processing.
    """
    assert isinstance(_seconds_to_time_stamp(seconds), str)
