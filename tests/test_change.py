"""Phase 5 endpoint tests: POST /api/change (bi-temporal change detection)."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend import vqa_service
from backend.app import app


def _jpg_bytes(color: str = "gray") -> bytes:
    image = Image.new("RGB", (16, 16), "white")
    ImageDraw.Draw(image).rectangle([2, 2, 12, 12], fill=color)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeMessage:
    content = "A gray square turned green between the two captures."


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def mock_completion(monkeypatch):
    monkeypatch.setattr(vqa_service, "_completion", lambda messages: _FakeCompletion())


def _post(client, *, question="What changed?", date1="2025-01-01", date2="2025-06-01"):
    return client.post(
        "/api/change",
        files=[
            ("image1", ("before.jpg", _jpg_bytes("gray"), "image/jpeg")),
            ("image2", ("after.jpg", _jpg_bytes("green"), "image/jpeg")),
        ],
        data={"question": question, "date1": date1, "date2": date2},
    )


def test_change_endpoint_returns_answer(client, mock_completion):
    """Two images + dates -> 200 with a non-empty answer."""
    response = _post(client)
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0


def test_change_endpoint_without_dates(client, mock_completion):
    """Dates are optional."""
    response = _post(client, date1=None, date2=None)
    assert response.status_code == 200


def test_change_endpoint_rejects_blank_question(client, mock_completion):
    response = _post(client, question="   ")
    assert response.status_code == 400


def test_change_endpoint_rejects_missing_image(client, mock_completion):
    """An empty second image -> 400."""
    response = client.post(
        "/api/change",
        files=[
            ("image1", ("before.jpg", _jpg_bytes("gray"), "image/jpeg")),
            ("image2", ("after.jpg", b"", "image/jpeg")),
        ],
        data={"question": "What changed?"},
    )
    assert response.status_code == 400


def test_change_endpoint_maps_model_failure_to_502(client, monkeypatch):
    def _boom(messages):
        raise vqa_service.VQAServiceError("model call failed")

    monkeypatch.setattr(vqa_service, "_completion", _boom)
    response = _post(client)
    assert response.status_code == 502
    assert "model call failed" in response.json()["detail"]


def test_change_sends_both_images_and_dates(client, monkeypatch):
    """The model prompt includes two image parts and the temporal context."""
    captured = {}

    def capture_completion(messages):
        captured["messages"] = messages
        return _FakeCompletion()

    monkeypatch.setattr(vqa_service, "_completion", capture_completion)
    assert _post(client).status_code == 200

    content = captured["messages"][0]["content"]
    image_parts = [c for c in content if c["type"] == "image_url"]
    text_parts = [c for c in content if c["type"] == "text"]
    assert len(image_parts) == 2
    assert text_parts[0]["text"].startswith("Two images of the same region")
    assert "2025-01-01" in text_parts[0]["text"]
    assert "2025-06-01" in text_parts[0]["text"]