import os


def test_file_exists():
    assert os.path.exists("saved_model.pb")
    assert os.path.exists("fingerprint.pb")
    assert os.path.exists("variables")
