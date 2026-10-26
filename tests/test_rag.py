import json
from unittest.mock import Mock, patch

import pytest
import requests

from src.rag import generate_answer, validate_citations
from src.retrieval import Hit


def test_citation_validation():
    assert validate_citations("Claim [S1] [S9] [S1]", {"S1"}) == (("S1",), ("S9",))


def test_no_evidence():
    assert generate_answer("question", []).status == "no_evidence"


def test_ollama_unavailable(chunks):
    with patch("src.rag.requests.post", side_effect=requests.ConnectionError):
        answer = generate_answer("question", [Hit(chunks[0], .8)])
    assert answer.status == "unavailable" and answer.text is None


@pytest.mark.parametrize("response,status", [
    (json.dumps({"claims": [{"text": "Networks learn.", "sources": ["S1"]}]}), "generated"),
    (json.dumps({"claims": [{"text": "Claim", "sources": ["S99"]}]}), "invalid_citations"),
    (json.dumps({"claims": [{"text": "Claim", "sources": []}]}), "uncited"),
    (json.dumps({"claims": []}), "abstained"),
    ("Unstructured assertion", "invalid_response"),
    (json.dumps({"insufficient_evidence": True, "claims": [{"text": "Untrusted mixed claim", "sources": ["S1"]}]}), "invalid_response"),
    (json.dumps({"claims": [{"text": "Claim [S9]", "sources": ["S1"]}]}), "invalid_citations"),
    ("", "unavailable")])
def test_model_responses(chunks, response, status):
    fake = Mock()
    fake.json.return_value = {"response": response}
    with patch("src.rag.requests.post", return_value=fake) as post:
        answer = generate_answer("question", [Hit(chunks[0], .8)])
        body = post.call_args.kwargs["json"]
        assert json.loads(body["prompt"])["evidence"][0]["chunk_id"] == "c1"
        assert "untrusted" in body["system"]
    assert answer.status == status
    if status in {"uncited", "invalid_citations", "unavailable", "invalid_response"}:
        assert answer.text is None
