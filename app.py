"""
app.py — Posture Classifier (No WebRTC)
Real-time via st_autorefresh + st.camera_input
Compatible: Streamlit Cloud, local, any network
"""
from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO

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
    "slouch": (
        "Duduk tegak dan tarik bahu ke belakang.",
        "Tempelkan pinggul ke sandaran kursi.",
    ),
    "hunch": (
        "Buka dada dan rilekskan bahu.",
        "Aktifkan otot perut agar tubuh tetap tegak.",
    ),
    "forward": (
        "Tarik kepala ke belakang agar telinga sejajar bahu.",
        "Naikkan posisi layar ke level mata.",
    ),
    "lean": (
        "Pusatkan berat badan dan duduk seimbang.",
        "Letakkan telapak kaki rata di lantai.",
    ),
    "tilt": (
        "Sejajarkan bahu dan jaga kepala tetap di tengah.",
        "Hindari miring terlalu lama ke satu sisi.",
    ),
}

# ─────────────────────────── Helpers ───────────────────────────


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_label(names: object, index: int) -> str:
    if isinstance(names, dict):
        return names.get(index, str(index))
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index])
    return str(index)


def topk_scores(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    k = max(1, min(k, scores.size))
    topk_idx = scores.argsort()[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in topk_idx]


def list_labels(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def infer_bad_labels(labels: list[str]) -> list[str]:
    bad = [l for l in labels if any(h in l.lower() for h in DEFAULT_BAD_LABEL_HINTS)]
    if bad:
        return bad
    return [l for l in labels if not any(h in l.lower() for h in DEFAULT_GOOD_LABEL_HINTS)]


def tips_for_label(label: str) -> list[str]:
    for key, tips in TIP_RULES.items():
        if key in label.lower():
            return list(tips)
    return list(DEFAULT_TIPS)


@st.cache_data
def alarm_audio_data_uri() -> str:
    duration, sample_rate, frequency = 0.25, 22050, 880.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (tone * np.iinfo(np.int16).max).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def play_alarm_audio() -> None:
    uri = alarm_audio_data_uri()
    st.components.v1.html(
        f'<audio autoplay="true"><source src="{uri}" type="audio/wav"></audio>',
        height=0,
    )


def run_inference(image: np.ndarray, model: YOLO, img_size: int) -> np.ndarray | None:
    """Run YOLO classify predict, return flat score array or None."""
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


def annotate_image(
    image: np.ndarray,
    scores: np.ndarray,
    model_names: object,
    bad_labels: set[str],
    alarm_threshold: float,
    conf_threshold: float,
    top_k: int,
) -> tuple[np.ndarray, str, float, bool]:
    """Draw overlays on image. Returns annotated image, top label, top conf, is_bad."""
    h, w = image.shape[:2]
    top1_idx = int(scores.argmax())
    top1_label = resolve_label(model_names, top1_idx)
    top1_conf = float(scores[top1_idx])
    is_bad = top1_label in bad_labels and top1_conf >= alarm_threshold

    out = image.copy()
    if is_bad:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(out, "WARNING: BAD POSTURE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

    line_h = 28
    n = min(top_k, scores.size)
    y = max(line_h, h - 10 - line_h * (n - 1))
    for idx, conf in topk_scores(scores, top_k):
        label = resolve_label(model_names, idx)
        color = (0, 255, 0) if conf >= conf_threshold else (0, 165, 255)
        cv2.putText(out, f"{label}: {conf:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28

    return out, top1_label, top1_conf, is_bad


# ─────────────────────────── UI ───────────────────────────

st.set_page_config(page_title="Posture Classifier", layout="wide")
st.title("🧍 Posture Classifier")
st.write("Kamera otomatis refresh tiap beberapa detik. Tidak butuh WebRTC.")

# ── Sidebar ──
weights = sorted([p.name for p in WEIGHTS_DIR.glob("*.pt")])
if not weights:
    st.error(f"No .pt weights found in {WEIGHTS_DIR}")
    st.stop()

selected_weight = st.sidebar.selectbox(
    "Model weights",
    weights,
    index=weights.index("best.pt") if "best.pt" in weights else 0,
)
img_size = st.sidebar.select_slider(
    "Image size", options=[160, 192, 224, 256, 288, 320], value=DEFAULT_IMG_SIZE
)
conf_threshold = st.sidebar.slider("Display threshold", 0.0, 1.0, 0.5, 0.01)
top_k = st.sidebar.slider("Top-K", 1, 5, 2)
alarm_threshold = st.sidebar.slider("Alarm threshold", 0.0, 1.0, DEFAULT_ALARM_THRESHOLD, 0.01)
refresh_interval = st.sidebar.select_slider(
    "Auto-refresh interval (ms)",
    options=[500, 1000, 1500, 2000, 3000],
    value=1500,
)

model = load_model(str(WEIGHTS_DIR / selected_weight))
label_options = list_labels(model.names)
bad_labels: set[str] = set(infer_bad_labels(label_options))

if bad_labels:
    st.sidebar.caption("Bad posture labels: " + ", ".join(sorted(bad_labels)))
else:
    st.sidebar.caption("Bad posture labels: none auto-detected.")

# ── Mode tabs ──
tab_live, tab_upload = st.tabs(["📷 Live (auto-refresh)", "🖼️ Upload / Snapshot"])

# ══════════════ TAB 1: Live mode ══════════════
with tab_live:
    st.info(
        "Klik **Take photo** lalu biarkan auto-refresh berjalan. "
        "Setiap interval, frame terakhir akan diklasifikasi ulang secara otomatis."
    )

    # Auto-refresh — only ticks when this tab is active (component is rendered)
    count = st_autorefresh(interval=refresh_interval, limit=None, key="live_refresh")

    photo = st.camera_input("Kamera", key=f"cam_{count}", label_visibility="collapsed")

    result_col, tips_col = st.columns([2, 1])

    with result_col:
        if photo is not None:
            file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                st.error("Gagal membaca frame kamera.")
            else:
                scores = run_inference(img, model, img_size)
                if scores is not None and scores.size:
                    annotated, top_label, top_conf, is_bad = annotate_image(
                        img, scores, model.names, bad_labels,
                        alarm_threshold, conf_threshold, top_k,
                    )
                    st.image(annotated, channels="BGR", use_container_width=True)

                    if is_bad:
                        st.error(f"⚠️ ALARM: postur buruk — **{top_label}** ({top_conf:.2f})")
                        play_alarm_audio()
                    else:
                        st.success(f"✅ Postur aman — **{top_label}** ({top_conf:.2f})")
                else:
                    st.warning("Model tidak mengembalikan skor. Coba ulang.")
        else:
            st.caption("Belum ada foto. Klik tombol kamera di atas.")

    with tips_col:
        # Show tips based on last known bad label stored in session state
        if photo is not None:
            file_bytes2 = np.asarray(bytearray(photo.getvalue()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            if img2 is not None:
                scores2 = run_inference(img2, model, img_size)
                if scores2 is not None and scores2.size:
                    top1_idx = int(scores2.argmax())
                    top1_label = resolve_label(model.names, top1_idx)
                    top1_conf = float(scores2[top1_idx])
                    is_bad2 = top1_label in bad_labels and top1_conf >= alarm_threshold
                    if is_bad2:
                        st.subheader("💡 Saran")
                        for tip in tips_for_label(top1_label):
                            st.markdown(f"- {tip}")

# ══════════════ TAB 2: Upload / Snapshot ══════════════
with tab_upload:
    camera_photo = st.camera_input("Ambil foto")
    uploaded = st.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])

    source = camera_photo if camera_photo is not None else uploaded

    if source is not None:
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Tidak bisa membaca file gambar.")
        else:
            scores = run_inference(image, model, img_size)
            if scores is None or not scores.size:
                st.error("Model tidak mengembalikan skor.")
            else:
                annotated, top_label, top_conf, is_bad = annotate_image(
                    image, scores, model.names, bad_labels,
                    alarm_threshold, conf_threshold, top_k,
                )

                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.image(annotated, channels="BGR", use_container_width=True, caption="Hasil")

                with col_info:
                    if is_bad:
                        st.error(f"⚠️ ALARM: postur buruk\n**{top_label}** ({top_conf:.2f})")
                        play_alarm_audio()
                        st.subheader("💡 Saran")
                        for tip in tips_for_label(top_label):
                            st.markdown(f"- {tip}")
                    else:
                        st.success(f"✅ Postur aman\n**{top_label}** ({top_conf:.2f})")

                    st.subheader("Top-K Scores")
                    for idx, conf in topk_scores(scores, top_k):
                        label = resolve_label(model.names, idx)
                        st.metric(label, f"{conf:.3f}")
