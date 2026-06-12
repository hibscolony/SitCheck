"""
app.py — Posture Classifier (True Real-Time, No WebRTC)
Mechanism:
  - Custom HTML component captures webcam frames via getUserMedia + Canvas
  - Every N ms, JS encodes frame as base64 JPEG and writes to a temp file
    via a small /upload_frame endpoint served by a background thread (Flask)
  - Streamlit main loop reads the temp file and runs YOLO inference
  - Results rendered in Streamlit UI

Actually simpler approach that truly works on Streamlit Cloud:
  - st.components.v1.html captures webcam as base64
  - Uses st.components bidirectional communication (component_value)
  - Each new frame triggers inference + UI update via session_state
"""
from __future__ import annotations

import base64
import io
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

try:
    from camera_input_live import camera_input_live
except ImportError:
    camera_input_live = None

# ─────────────────────────── Constants ───────────────────────────

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_IMG_SIZE = 224
DEFAULT_ALARM_THRESHOLD = 0.7
DEFAULT_BAD_LABEL_HINTS = ("bad", "slouch", "hunch", "forward", "lean", "tilt")
DEFAULT_GOOD_LABEL_HINTS = ("good", "normal", "upright", "correct", "ok", "proper")
DEFAULT_TIPS = (
    "Duduk tegak dengan punggung tersandar.",
    "Jaga telinga sejajar dengan bahu.",
    "Rilekskan bahu dan hindari membungkuk.",
    "Atur tinggi layar sejajar dengan mata.",
)
TIP_RULES = {
    "slouch": ("Duduk tegak dan tarik bahu ke belakang.", "Tempelkan pinggul ke sandaran kursi."),
    "hunch": ("Buka dada dan rilekskan bahu.", "Aktifkan otot perut agar tubuh tetap tegak."),
    "forward": ("Tarik kepala ke belakang agar telinga sejajar bahu.", "Naikkan posisi layar ke level mata."),
    "lean": ("Pusatkan berat badan dan duduk seimbang.", "Letakkan telapak kaki rata di lantai."),
    "tilt": ("Sejajarkan bahu dan jaga kepala tetap di tengah.", "Hindari miring terlalu lama ke satu sisi."),
}

# ─────────────────────────── Helpers ───────────────────────────

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_label(names, index: int) -> str:
    if isinstance(names, dict):
        return names.get(index, str(index))
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index])
    return str(index)


def topk_scores(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    k = max(1, min(k, scores.size))
    idx = scores.argsort()[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]


def list_labels(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def infer_bad_labels(labels: list[str]) -> list[str]:
    bad = [l for l in labels if any(h in l.lower() for h in DEFAULT_BAD_LABEL_HINTS)]
    return bad if bad else [l for l in labels if not any(h in l.lower() for h in DEFAULT_GOOD_LABEL_HINTS)]


def tips_for_label(label: str) -> list[str]:
    for key, tips in TIP_RULES.items():
        if key in label.lower():
            return list(tips)
    return list(DEFAULT_TIPS)


@st.cache_data
def alarm_audio_b64() -> str:
    sr, freq, dur = 22050, 880.0, 0.25
    t = np.linspace(0, dur, int(sr * dur), False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t) * np.iinfo(np.int16).max).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_inference(image: np.ndarray, model: YOLO, img_size: int) -> np.ndarray | None:
    results = model.predict(image, imgsz=img_size, verbose=False)
    if not results:
        return None
    probs = getattr(results[0], "probs", None)
    if probs is None or getattr(probs, "data", None) is None:
        return None
    scores = probs.data
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    return np.asarray(scores).reshape(-1)


def annotate_image(img, scores, names, bad_labels, alarm_thr, conf_thr, top_k):
    h, w = img.shape[:2]
    top1 = int(scores.argmax())
    label = resolve_label(names, top1)
    conf = float(scores[top1])
    is_bad = label in bad_labels and conf >= alarm_thr
    out = img.copy()
    if is_bad:
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 0, 255), 6)
        cv2.putText(out, "WARNING: BAD POSTURE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    lh = 28
    y = max(lh, h - 10 - lh * (min(top_k, scores.size) - 1))
    for i, c in topk_scores(scores, top_k):
        lbl = resolve_label(names, i)
        color = (0, 255, 0) if c >= conf_thr else (0, 165, 255)
        cv2.putText(out, f"{lbl}: {c:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += lh
    return out, label, conf, is_bad


# ─────────────────────────── Page ───────────────────────────

st.set_page_config(page_title="Posture Classifier", layout="wide")
st.title("🧍 Posture Classifier")

# ── Sidebar ──
weights = sorted([p.name for p in WEIGHTS_DIR.glob("*.pt")])
if not weights:
    st.error(f"No .pt weights in {WEIGHTS_DIR}"); st.stop()

sel_w = st.sidebar.selectbox("Model weights", weights,
    index=weights.index("best.pt") if "best.pt" in weights else 0)
img_size    = st.sidebar.select_slider("Image size",
    options=[160,192,224,256,288,320], value=DEFAULT_IMG_SIZE)
conf_thr    = st.sidebar.slider("Display threshold", 0.0, 1.0, 0.5, 0.01)
top_k       = st.sidebar.slider("Top-K", 1, 5, 2)
alarm_thr   = st.sidebar.slider("Alarm threshold", 0.0, 1.0, DEFAULT_ALARM_THRESHOLD, 0.01)
interval_ms = st.sidebar.select_slider("Capture interval",
    options=[500, 750, 1000, 1500, 2000], value=1000,
    format_func=lambda x: f"{x} ms")

model = load_model(str(WEIGHTS_DIR / sel_w))
labels = list_labels(model.names)
bad_labels: set[str] = set(infer_bad_labels(labels))
st.sidebar.caption("Bad labels: " + (", ".join(sorted(bad_labels)) or "none detected"))

alarm_b64 = alarm_audio_b64()

# ─────────────────────────── Session state ───────────────────────────
for k, v in [("frame_b64", None), ("last_label", ""), ("last_conf", 0.0),
             ("last_is_bad", False), ("running", False), ("frame_idx", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────── Layout ───────────────────────────

tab_live, tab_upload = st.tabs(["📷 Live Camera", "🖼️ Upload / Snapshot"])

with tab_live:
    if camera_input_live is None:
        st.error(
            "Paket **streamlit-camera-input-live** belum terpasang.\n\n"
            "Tambahkan `streamlit-camera-input-live` ke `requirements.txt` "
            "lalu jalankan `pip install streamlit-camera-input-live`."
        )
    else:

        @st.fragment
        def live_camera_fragment():
            col_cam, col_result = st.columns([3, 2])

            with col_cam:
                # Returns a fresh BytesIO JPEG every `interval_ms` automatically
                # (no manual Start/Stop wiring needed, no WebRTC).
                # Wrapped in a fragment so only THIS block reruns on each new
                # frame, instead of the whole page (no more full-page "kedip").
                image_buf = camera_input_live(
                    debounce=interval_ms,
                    height=560,
                    width=860,
                    key="posture_live_cam",
                    show_controls=True,
                )

            with col_result:
                result_box  = st.empty()
                status_box  = st.empty()
                tips_box    = st.empty()
                topk_box    = st.empty()

                # Process incoming frame
                if image_buf is not None:
                    try:
                        img_bytes = image_buf.getvalue()
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                        if img is not None:
                            st.session_state.frame_idx += 1
                            scores = run_inference(img, model, img_size)
                            if scores is not None and scores.size:
                                annotated, label, conf, is_bad = annotate_image(
                                    img, scores, model.names, bad_labels,
                                    alarm_thr, conf_thr, top_k
                                )

                                result_box.image(annotated, channels="BGR", use_column_width=True)

                                if is_bad:
                                    status_box.error(f"⚠️ **{label}** — {conf:.2f}")
                                    tips_box.info("💡 Saran:\n" + "\n".join(
                                        f"- {t}" for t in tips_for_label(label)
                                    ))
                                    # Play alarm audio (nonce forces iframe reload -> autoplay each time)
                                    components.html(
                                        f'<!-- frame {st.session_state.frame_idx} -->'
                                        f'<audio autoplay><source src="data:audio/wav;base64,{alarm_b64}" type="audio/wav"></audio>',
                                        height=0
                                    )
                                else:
                                    status_box.success(f"✅ **{label}** — {conf:.2f}")
                                    tips_box.empty()

                                topk_lines = "\n".join(
                                    f"`{resolve_label(model.names, i)}` {c:.3f}"
                                    for i, c in topk_scores(scores, top_k)
                                )
                                topk_box.markdown(f"**Top-{top_k}:**\n{topk_lines}")

                    except Exception as e:
                        status_box.warning(f"Frame error: {e}")
                else:
                    status_box.info("👈 Klik **Start capturing** di panel kamera untuk memulai.")

        live_camera_fragment()


# ──────────────────────────────────────────
with tab_upload:
    camera_photo = st.camera_input("Ambil foto")
    uploaded     = st.file_uploader("Atau upload gambar", type=["jpg","jpeg","png"])
    source = camera_photo if camera_photo is not None else uploaded

    if source is not None:
        raw   = np.asarray(bytearray(source.read()), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Tidak bisa membaca file.")
        else:
            scores = run_inference(image, model, img_size)
            if scores is None or not scores.size:
                st.error("Model tidak mengembalikan skor.")
            else:
                annotated, label, conf, is_bad = annotate_image(
                    image, scores, model.names, bad_labels,
                    alarm_thr, conf_thr, top_k
                )
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.image(annotated, channels="BGR", use_column_width=True)
                with c2:
                    if is_bad:
                        st.error(f"⚠️ **{label}** ({conf:.2f})")
                        components.html(
                            f'<audio autoplay><source src="data:audio/wav;base64,{alarm_b64}" type="audio/wav"></audio>',
                            height=0
                        )
                        st.subheader("💡 Saran")
                        for t in tips_for_label(label):
                            st.markdown(f"- {t}")
                    else:
                        st.success(f"✅ **{label}** ({conf:.2f})")
                    st.subheader(f"Top-{top_k}")
                    for i, c in topk_scores(scores, top_k):
                        st.metric(resolve_label(model.names, i), f"{c:.3f}")
