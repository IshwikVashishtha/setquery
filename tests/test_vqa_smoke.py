"""Smoke tests for the Phase 1 VQA service.

Runs against a generated sample image. ``answer_question`` is exercised two ways:
  * mocked model call — always runs, keeps CI green without an ``HF_TOKEN``;
  * real model call — skipped unless ``HF_TOKEN`` is set in the environment.
"""

import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend import vqa_service
from backend.app import app
from backend.vqa_service import VQAServiceError, answer_question


def _sample_image() -> Image.Image:
    """A real, non-placeholder 64x64 image with visible structure."""
    image = Image.new("RGB", (64, 64), "skyblue")
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 28, 28], fill="gray")   # one "building"
    draw.rectangle([36, 20, 56, 40], fill="gray")  # another
    draw.rectangle([20, 36, 44, 56], fill="green")  # a "field"
    return image


class _FakeMessage:
    content = "There appear to be a few rectangular structures and a green field."


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]


@pytest.fixture()
def mock_completion(monkeypatch):
    """Replace the model call with a fake completion for deterministic tests."""
    monkeypatch.setattr(vqa_service, "_completion", lambda messages: _FakeCompletion())


@pytest.fixture()
def client():
    """A TestClient against the FastAPI app (does not bind a port)."""
    with TestClient(app) as test_client:
        yield test_client


def _image_bytes(image: Image.Image | None = None) -> bytes:
    buffer = BytesIO()
    (image or _sample_image()).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_answer_question_mocked_returns_text(mock_completion):
    """With the model call mocked, returns a non-empty string."""
    answer = answer_question(_sample_image(), "What is in this image?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_answer_question_raises_when_token_missing(monkeypatch):
    """Without HF_TOKEN (and no mock), raises a clear VQAServiceError."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(VQAServiceError):
        answer_question(_sample_image(), "What is in this image?")


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set; running the real model call requires a token.",
)
def test_answer_question_real_returns_text():
    """Against the hosted VLM, returns a coherent non-empty answer."""
    answer = answer_question(_sample_image(), "What is in this image?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# --- Endpoint contract (Design.md §4) ---


def test_health(client):
    """GET /health -> {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vqa_endpoint_returns_answer(client, mock_completion):
    """POST /api/vqa with a valid image + question -> 200 {"answer": str}."""
    response = client.post(
        "/api/vqa",
        files={"image": ("image.jpg", _image_bytes(), "image/jpeg")},
        data={"question": "What is in this image?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0


def test_vqa_endpoint_rejects_invalid_image(client, mock_completion):
    """A non-image upload -> 400."""
    response = client.post(
        "/api/vqa",
        files={"image": ("not-an-image.txt", b"this is not an image", "text/plain")},
        data={"question": "What is in this image?"},
    )
    assert response.status_code == 400


def test_vqa_endpoint_rejects_empty_question(client, mock_completion):
    """A blank question -> 400."""
    response = client.post(
        "/api/vqa",
        files={"image": ("image.jpg", _image_bytes(), "image/jpeg")},
        data={"question": "   "},
    )
    assert response.status_code == 400


def test_vqa_endpoint_maps_model_failure_to_502(client, monkeypatch):
    """When the model call raises, the endpoint responds 502, not 200."""

    def _boom(messages):
        raise vqa_service.VQAServiceError("model call failed")

    monkeypatch.setattr(vqa_service, "_completion", _boom)
    response = client.post(
        "/api/vqa",
        files={"image": ("image.jpg", _image_bytes(), "image/jpeg")},
        data={"question": "What is in this image?"},
    )
    assert response.status_code == 502
    assert "model call failed" in response.json()["detail"]