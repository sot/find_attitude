import agasc
import pytest


@pytest.fixture(autouse=True)
def proseco_agasc_rc(monkeypatch):
    agasc_file = agasc.get_agasc_filename("proseco_agasc_*", allow_rc=True)
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)


@pytest.fixture()
def proseco_agasc_1p7(monkeypatch):
    agasc_file = agasc.get_agasc_filename("proseco_agasc_*", version="1p7")
    monkeypatch.setenv("AGASC_HDF5_FILE", agasc_file)
