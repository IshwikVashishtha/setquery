"""Gradio frontend for the VQA agent.

Upload an image, type a question, get a text answer from the FastAPI backend.
Run with ``python frontend/app.py`` while ``uvicorn backend.app:app`` is up.
"""

import os

import gradio as gr
import httpx
from PIL import Image

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def ask_vqa(image: Image.Image, question: str) -> str:
    """Send ``image`` + ``question`` to the backend and return its answer."""
    if image is None or not question.strip():
        return "Please upload an image and type a question."

    buffer = _image_to_bytes(image)
    files = {"image": ("image.jpg", buffer, "image/jpeg")}
    data = {"question": question}

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{BACKEND_URL}/api/vqa", files=files, data=data
            )
    except httpx.HTTPError as exc:
        return f"Could not reach the backend at {BACKEND_URL}: {exc}"

    if response.status_code != 200:
        detail = response.json().get("detail", "unknown error")
        return f"Backend error {response.status_code}: {detail}"
    return response.json()["answer"]


def _image_to_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to JPEG bytes for the multipart upload."""
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def build_app() -> gr.Blocks:
    """Assemble and return the Gradio UI."""
    with gr.Blocks(title="Remote Sensing VQA Agent") as demo:
        gr.Markdown(
            "# Remote Sensing VQA Agent\n"
            "Upload a satellite/aerial image and ask a question about it."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Image")
            question_input = gr.Textbox(
                label="Question", placeholder="e.g. How many buildings are visible?"
            )
        answer_output = gr.Textbox(label="Answer", lines=4)
        submit = gr.Button("Ask")
        submit.click(ask_vqa, inputs=[image_input, question_input], outputs=answer_output)
    return demo


if __name__ == "__main__":
    build_app().launch()