"""Gradio frontend for the VQA agent.

Two tabs:
  * **Ask** — upload an image, ask a plain-English question, get a text answer
    (Phases 1-5).
  * **Segment & Map** — upload an image + an object class, get the detected
    bounding boxes drawn on the image; GeoTIFF uploads additionally show an
    interactive Leaflet map (Folium) with the image and the boxes overlaid at
    their real-world locations (Phase 6).

Run with ``python frontend/app.py`` while ``uvicorn backend.app:app`` is up.
"""

import base64
import io
import os

import gradio as gr
import httpx
from PIL import Image, ImageDraw, ImageFont

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_BOX_COLOR = "#ff2020"


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


def ground_objects(image: Image.Image, geotiff_file, query: str):
    """Send the image (or GeoTIFF) + ``query`` to ``/api/ground``.

    ``geotiff_file`` takes priority so a raw GeoTIFF reaches the backend
    byte-identical (Gradio's image upload would flatten it to JPEG and lose the
    geo-referencing the map overlay needs). Returns
    ``(annotated_image, summary_text, map_html)``; the summary lists every
    detected box with pixel (and, for GeoTIFFs, real-world) coordinates.
    """
    if image is None and geotiff_file is None:
        return image, "Please upload an image or a GeoTIFF and type an object to find.", None
    if not query.strip():
        return image, "Please type an object to find.", None

    if geotiff_file is not None:
        filename, file_bytes = _read_upload(geotiff_file)
        ext = filename.rsplit(".", 1)[-1].lower()
        files = {"image": (f"upload.{ext}", file_bytes, "application/octet-stream")}
    else:
        files = {"image": ("image.jpg", _image_to_bytes(image), "image/jpeg")}
    data = {"query": query}

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{BACKEND_URL}/api/ground", files=files, data=data
            )
    except httpx.HTTPError as exc:
        return image, f"Could not reach the backend at {BACKEND_URL}: {exc}", None

    if response.status_code != 200:
        detail = response.json().get("detail", "unknown error")
        return image, f"Backend error {response.status_code}: {detail}", None

    payload = response.json()
    preview = _draw_boxes(payload["preview"], payload["boxes"])
    summary = _summarize_boxes(payload["boxes"], payload.get("geo_bounds"))
    return preview, summary, payload.get("map_html")


def _draw_boxes(preview_b64: str, boxes: list[dict]) -> Image.Image:
    """Draw the ground-truth-pixel boxes onto the preview PNG."""
    image = Image.open(io.BytesIO(base64.b64decode(preview_b64))).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()
    for box in boxes:
        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline=_BOX_COLOR,
            width=3,
        )
        draw.rectangle(
            [box["xmin"], box["ymax"] - 18, box["xmax"], box["ymax"]],
            fill=_BOX_COLOR,
        )
        draw.text((box["xmin"] + 4, box["ymax"] - 16), box["label"], fill="white", font=font)
    return image


def _summarize_boxes(boxes: list[dict], geo_bounds) -> str:
    """Human-readable list of detections with pixel and geo coordinates."""
    if not boxes:
        return "No boxes were detected for this query."
    lines = [f"{len(boxes)} detection(s):"]
    for i, box in enumerate(boxes, 1):
        conf = f" ({box['confidence']:.0%} conf)" if box.get("confidence") is not None else ""
        line = (
            f"  {i}. {box['label']}{conf}  "
            f"pixels ({box['xmin']}, {box['ymin']}) -> ({box['xmax']}, {box['ymax']})"
        )
        if box.get("lonlat"):
            lon, lat = box["lonlat"]
            line += f"  |  geo center ({lat:.6f}°, {lon:.6f}°)"
        lines.append(line)
    if geo_bounds:
        lines.append(f"  Map extent: {geo_bounds} (lat/lon)")
    return "\n".join(lines)


def _read_upload(upload):
    """Read a Gradio file upload (path or file-like) into ``(name, bytes)``."""
    if hasattr(upload, "read"):
        return getattr(upload, "name", "upload"), upload.read()
    filename = str(upload)
    with open(filename, "rb") as fh:
        return filename, fh.read()


def _image_to_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to JPEG bytes for the multipart upload."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def build_app() -> gr.Blocks:
    """Assemble and return the Gradio UI."""
    with gr.Blocks(title="Remote Sensing VQA Agent") as demo:
        gr.Markdown(
            "# Remote Sensing VQA Agent\n"
            "Upload a satellite/aerial image, ask a question, or locate objects "
            "on a map."
        )
        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Image")
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. How many buildings are visible?",
                    )
                answer_output = gr.Textbox(label="Answer", lines=4)
                submit = gr.Button("Ask")
                submit.click(
                    ask_vqa, inputs=[image_input, question_input], outputs=answer_output
                )

            with gr.Tab("Segment & Map"):
                ground_markdown = gr.Markdown(
                    "Type the object to locate, e.g. `buildings`. Upload a "
                    "**JPG/PNG** in the image box, or attach a **GeoTIFF** "
                    "below it to also get a Leaflet map with the boxes overlaid "
                    "at their real-world locations."
                )
                with gr.Row():
                    ground_image_input = gr.Image(
                        type="pil", label="Image (JPG/PNG)"
                    )
                    ground_file_input = gr.File(
                        label="Or upload a GeoTIFF (.tif)",
                        file_types=[".tif", ".tiff"],
                    )
                ground_query_input = gr.Textbox(
                    label="Object to find", placeholder="e.g. buildings, water"
                )
                ground_output = gr.Image(label="Detections", type="pil")
                ground_coords = gr.Textbox(label="Detection coordinates", lines=6)
                ground_map = gr.HTML(label="Map overlay (GeoTIFF only)")
                ground_submit = gr.Button("Locate")
                ground_submit.click(
                    ground_objects,
                    inputs=[ground_image_input, ground_file_input, ground_query_input],
                    outputs=[ground_output, ground_coords, ground_map],
                )
    return demo


if __name__ == "__main__":
    build_app().launch()