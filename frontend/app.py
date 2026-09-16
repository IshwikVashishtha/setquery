"""Gradio frontend for the VQA agent.

Two tabs:
  * **Chat** -- upload a JPG/PNG (or a GeoTIFF, which renders a real preview),
    then ask plain-English questions in a running chat. The image stays
    attached as context across turns; uploading a new image resets the
    conversation and replaces the context. Each question is posted to
    ``/api/analyze`` and the answer optionally lists which backend tools were
    used (Phases 1-5 + tool loop).
  * **Segment & Map** -- upload an image + an object class, get the detected
    bounding boxes drawn on the image; GeoTIFF uploads additionally show an
    interactive Leaflet map (Folium) with the image and the boxes overlaid at
    their real-world locations (Phase 6).

Run with ``python frontend/app.py`` while ``uvicorn backend.app:app`` is up.
"""

import base64
import io
import os
import sys
from pathlib import Path

# Allow running as ``python frontend/app.py`` from the repo root: make the
# repo root importable so ``backend.grounding`` (band-stretched preview) is
# reachable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import httpx
from PIL import Image, ImageDraw, ImageFont

from backend.grounding import render_rgb_preview

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_BOX_COLOR = "#ff2020"
_NO_IMAGE_HINT = "Attach an image first, then ask your question."


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def _tool_label(tool: dict) -> str:
    """Compact ``server/name`` label for an entry in ``tools_used``."""
    server, name = tool.get("server"), tool.get("name")
    if server and name:
        return f"{server}/{name}"
    return name or server or "?"


def _ask_analyze(image: Image.Image, geotiff_file, question: str) -> str:
    """POST ``image`` (or GeoTIFF) + ``question`` to ``/api/analyze``.

    ``geotiff_file`` (a raw .tif upload) takes priority: its bytes reach the
    backend byte-identical so all bands reach the index computation. The image
    box is then just the visual preview.  Returns a markdown string ready for
    the Chatbot, including a compact list of tools used when provided.
    """
    if geotiff_file is not None:
        filename, file_bytes = _read_upload(geotiff_file)
        ext = filename.rsplit(".", 1)[-1].lower()
        files = {"image": (f"upload.{ext}", file_bytes, "application/octet-stream")}
    elif image is not None:
        files = {"image": ("image.jpg", _image_to_bytes(image), "image/jpeg")}
    else:
        return _NO_IMAGE_HINT

    data = {"question": question}

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{BACKEND_URL}/api/analyze", files=files, data=data
            )
    except httpx.HTTPError as exc:
        return f"Could not reach the backend at {BACKEND_URL}: {exc}"

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "unknown error")
        except ValueError:
            detail = response.text[:200] or "invalid response"
        return f"Backend error {response.status_code}: {detail}"

    payload = response.json()
    answer = (payload.get("answer") or "").strip() or "(no answer returned)"
    tools = payload.get("tools_used") or []
    if tools:
        names = ", ".join(dict.fromkeys(_tool_label(t) for t in tools))
        answer += f"\n\n_Used tools:_ {names}"
    return answer


def chat_send(message: str, history: list, image, tif_file) -> tuple[list, str]:
    """Append a user turn + assistant answer to the Chatbot history.

    Reads both upload components so the raw GeoTIFF (if present) wins over
    the image box.  Returns ``(history, "")`` with the textbox cleared.
    """
    history = list(history or [])
    question = str(message or "").strip()
    if not question:
        return history, ""

    if tif_file is None and image is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": _NO_IMAGE_HINT})
        return history, ""

    history.append({"role": "user", "content": question})
    history.append(
        {"role": "assistant", "content": _ask_analyze(image, tif_file, question)}
    )
    return history, ""


def _reset_chat_on_upload() -> tuple[list, None]:
    """A new JPG/PNG upload replaces the context: reset chat, drop any tif."""
    return [], None


def _on_tif_upload(tif_file):
    """A new GeoTIFF becomes the context: show its preview and reset the chat.

    Returns ``(preview, [])`` so the image box renders the real band-stretched
    scene and the conversation starts fresh.  Defensive ``gr.skip()`` outputs
    for the (never fired) cleared-file case keep an existing preview intact.
    """
    if tif_file is None:
        return gr.skip(), gr.skip()
    return _tif_preview(tif_file), []


# ---------------------------------------------------------------------------
# GeoTIFF preview renderer
# ---------------------------------------------------------------------------

def _tif_preview(geotiff_file) -> Image.Image | None:
    """Render a GeoTIFF as a viewable, band-stretched image for the preview box.

    Browsers can't display TIFF and PIL can't open many GeoTIFFs (multi-band,
    uint16, SAR int16), which is why a raw .tif upload into ``gr.Image`` shows
    a placeholder.  This reuses the backend's percentile-stretched RGB renderer
    so the user sees the actual scene instead.
    """
    if geotiff_file is None:
        return None
    try:
        preview, _w, _h = render_rgb_preview(str(geotiff_file))
        return preview
    except Exception as exc:
        # Diagnosable placeholder instead of a silent blank.
        img = Image.new("RGB", (640, 480), (32, 32, 32))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((16, 16), f"Could not render GeoTIFF preview:", fill=(255, 255, 255), font=font)
        draw.text((16, 40), f"{type(exc).__name__}: {str(exc)[:80]}", fill=(255, 120, 120), font=font)
        return img


# ---------------------------------------------------------------------------
# Segment & Map (Phase 6) -- kept verbatim
# ---------------------------------------------------------------------------

def ground_objects(image: Image.Image, geotiff_file, query: str):
    """Send the image (or GeoTIFF) + ``query`` to ``/api/ground``.

    ``geotiff_file`` takes priority so a raw GeoTIFF reaches the backend
    byte-identical (Gradio's image upload would flatten it to JPEG and lose the
    geo-referencing the map overlay needs).  Returns
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble and return the Gradio UI."""
    with gr.Blocks(title="Remote Sensing VQA Agent") as demo:
        gr.Markdown(
            "# Remote Sensing VQA Agent\n"
            "Upload a satellite/aerial image, then ask it questions in the chat "
            "below -- or locate objects on a map in the second tab."
        )
        with gr.Tabs():
            # ---- Chat tab ----
            with gr.Tab("Chat"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Image (JPG/PNG)")
                    tif_file_input = gr.File(
                        label="Or upload a GeoTIFF (.tif)",
                        file_types=[".tif", ".tiff"],
                    )
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Conversation",
                    height=420,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Ask about the image",
                        placeholder="e.g. How many buildings are visible?",
                        scale=5,
                    )
                    send_btn = gr.Button("Send", scale=1)

                # Enter-to-send
                chat_input.submit(
                    chat_send,
                    inputs=[chat_input, chatbot, image_input, tif_file_input],
                    outputs=[chatbot, chat_input],
                )
                # Button send
                send_btn.click(
                    chat_send,
                    inputs=[chat_input, chatbot, image_input, tif_file_input],
                    outputs=[chatbot, chat_input],
                )
                # New JPG/PNG replaces context
                image_input.upload(
                    _reset_chat_on_upload,
                    outputs=[chatbot, tif_file_input],
                )
                # New GeoTIFF replaces context (preview + chat reset)
                tif_file_input.upload(
                    _on_tif_upload,
                    inputs=[tif_file_input],
                    outputs=[image_input, chatbot],
                )

            # ---- Segment & Map tab (unchanged) ----
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
                ground_file_input.change(
                    _tif_preview, inputs=[ground_file_input], outputs=[ground_image_input],
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
    build_app().launch(share=True)
