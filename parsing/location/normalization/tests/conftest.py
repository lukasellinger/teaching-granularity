import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Normalization tests must not contact Wikidata")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
