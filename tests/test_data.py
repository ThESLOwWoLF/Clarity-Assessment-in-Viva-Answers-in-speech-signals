import pytest

from src.data import normalize_label, recording_id, to_wav_filename


@pytest.mark.parametrize("raw,want", [
    ("High", "high"), (" h ", "high"), ("Medium", "medium"), ("med", "medium"), ("m", "medium"),
    ("LOW", "low"), ("l", "low"), (0.9, "high"), (0.66, "high"), (0.5, "medium"), (0.33, "medium"),
    (0.1, "low"), ("0.7", "high"), (float("nan"), None), (None, None), ("unclear", None)])
def test_normalize_label(raw, want):
    assert normalize_label(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("107s1", "107_s1.wav"), ("107_s1", "107_s1.wav"), ("107 s 12", "107_s12.wav"), ("107_S3.WAV", "107_s3.wav"),
    ("107", None), ("abc_s1", None), ("107_s1_s2", None)])
def test_to_wav_filename(raw, want):
    assert to_wav_filename(raw) == want


@pytest.mark.parametrize("path,want", [
    ("wav files/107_s3.wav", "107"), ("107_s12.wav", "107"), (r"C:\data\wav files\2_s1.wav", "2")])
def test_recording_id(path, want):
    assert recording_id(path.replace("\\", "/")) == want
