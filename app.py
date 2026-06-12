"""
app.py — Posture Classifier (True Real-Time, No WebRTC)

Fix summary:
  - Tidak kedip: fragment hanya rerun saat ada frame baru (benar),
    tapi TIDAK menyimpan st.empty() di session_state (stale index crash).
    Sebaliknya, hasil inferensi disimpan sebagai DATA di session_state,
    lalu UI dirender ulang dari data tersebut setiap fragment rerun.
  - Bad info stabil: majority-vote rolling window sebelum label dianggap bad.
  - Alarm tidak spam: cooldown berbasis frame_idx.
"""
from __future__ import annotations

import base64
import io
import wave
from collections import deque
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

SMOOTH_WINDOW = 3

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
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
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


def smoothed_is_bad(is_bad_raw: bool) -> bool:
    """Majority-vote atas rolling window → cegah flip-flop 1 frame."""
    buf: deque = st.session_state.bad_history
    buf.append(is_bad_raw)
    return sum(buf) > len(buf) / 2


# ─────────────────────────── Page ───────────────────────────

st.set_page_config(page_title="SitCheck", layout="wide")
st.title("🪑 SitCheck")

# ── Sidebar ──
weights = sorted([p.name for p in WEIGHTS_DIR.glob("*.pt")])
if not weights:
    st.error(f"No .pt weights in {WEIGHTS_DIR}"); st.stop()

sel_w       = st.sidebar.selectbox("Model weights", weights,
                index=weights.index("best.pt") if "best.pt" in weights else 0)
img_size    = st.sidebar.select_slider("Image size",
                options=[160, 192, 224, 256, 288, 320], value=DEFAULT_IMG_SIZE)
conf_thr    = st.sidebar.slider("Display threshold", 0.0, 1.0, 0.5, 0.01)
top_k       = st.sidebar.slider("Top-K", 1, 5, 2)
alarm_thr   = st.sidebar.slider("Alarm threshold", 0.0, 1.0, DEFAULT_ALARM_THRESHOLD, 0.01)
interval_ms = st.sidebar.select_slider("Capture interval",
                options=[500, 750, 1000, 1500, 2000], value=1000,
                format_func=lambda x: f"{x} ms")
smooth_win  = st.sidebar.slider("Smoothing window", 1, 7, SMOOTH_WINDOW,
                help="Jumlah frame untuk majority-vote sebelum bad posture dikonfirmasi.")

model      = load_model(str(WEIGHTS_DIR / sel_w))
labels     = list_labels(model.names)
bad_labels: set[str] = set(infer_bad_labels(labels))
st.sidebar.caption("Bad labels: " + (", ".join(sorted(bad_labels)) or "none detected"))

alarm_b64 = alarm_audio_b64()

# ─────────────────────────── Session state ───────────────────────────
_ss_defaults: dict = {
    "frame_idx":    0,
    "last_alarm":   -999,
    "bad_history":  deque(maxlen=smooth_win),
    # Hasil inferensi terakhir disimpan sebagai data (bukan widget)
    "last_annotated": None,   # np.ndarray BGR
    "last_label":     "",
    "last_conf":      0.0,
    "last_is_bad":    False,
    "last_topk":      [],     # list[tuple[int,float]]
    "play_alarm":     False,  # flag: bunyikan alarm pada render ini
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync maxlen smoothing window jika slider berubah
if st.session_state.bad_history.maxlen != smooth_win:
    old = list(st.session_state.bad_history)
    st.session_state.bad_history = deque(old[-smooth_win:], maxlen=smooth_win)

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
            # ── Baris 1: kamera full-width ──
            image_buf = camera_input_live(
                debounce=interval_ms,
                height=480,
                key="posture_live_cam",
                show_controls=True,
            )

            # ── Proses frame baru (hanya jika ada input) ──
            if image_buf is not None:
                try:
                    img_bytes = image_buf.getvalue()
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if img is not None:
                        st.session_state.frame_idx += 1
                        fidx = st.session_state.frame_idx

                        scores = run_inference(img, model, img_size)
                        if scores is not None and scores.size:
                            annotated, label, conf, is_bad_raw = annotate_image(
                                img, scores, model.names, bad_labels,
                                alarm_thr, conf_thr, top_k
                            )
                            is_bad = smoothed_is_bad(is_bad_raw)

                            # Encode ke JPEG bytes — lebih aman di session_state
                            # daripada raw numpy array (hindari serialization error)
                            ok, buf_enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            st.session_state.last_annotated = buf_enc.tobytes() if ok else None
                            st.session_state.last_label     = label
                            st.session_state.last_conf      = conf
                            st.session_state.last_is_bad    = is_bad
                            st.session_state.last_topk      = topk_scores(scores, top_k)

                            alarm_cooldown = max(3, smooth_win)
                            if is_bad and fidx - st.session_state.last_alarm >= alarm_cooldown:
                                st.session_state.last_alarm = fidx
                                st.session_state.play_alarm = True
                            else:
                                st.session_state.play_alarm = False

                except Exception as e:
                    st.session_state.last_label = f"error: {e}"

            # ── Baris 2: hasil annotasi full-width ──
            if st.session_state.last_annotated is not None:
                st.image(st.session_state.last_annotated,
                         use_column_width=True)

                # ── Baris 3: status | saran | top-k dalam 3 kolom ──
                col_status, col_tips, col_topk = st.columns([1, 2, 1])

                with col_status:
                    if st.session_state.last_is_bad:
                        st.error(
                            f"⚠️ **Postur Buruk**\n\n"
                            f"`{st.session_state.last_label}` — {st.session_state.last_conf:.2f}"
                        )
                        if st.session_state.play_alarm:
                            fidx = st.session_state.frame_idx
                            components.html(
                                f'<!-- nonce:{fidx} -->'
                                f'<audio autoplay>'
                                f'<source src="data:audio/wav;base64,{alarm_b64}"'
                                f' type="audio/wav"></audio>',
                                height=0,
                            )
                    else:
                        st.success(
                            f"✅ **Postur Baik**\n\n"
                            f"`{st.session_state.last_label}` — {st.session_state.last_conf:.2f}"
                        )

                with col_tips:
                    if st.session_state.last_is_bad:
                        tips = tips_for_label(st.session_state.last_label)
                        st.info("💡 **Saran:**\n" + "\n".join(f"- {t}" for t in tips))
                    else:
                        st.info("💡 **Pertahankan postur yang baik!**\n- Tetap rileks dan tegak.\n- Istirahat setiap 30 menit.")

                with col_topk:
                    if st.session_state.last_topk:
                        st.markdown(f"**Top-{top_k} Skor:**")
                        for i, c in st.session_state.last_topk:
                            lbl = resolve_label(model.names, i)
                            st.metric(label=lbl, value=f"{c:.3f}")

            elif image_buf is None:
                st.info("👈 Klik **Start capturing** di panel kamera untuk memulai.")

        live_camera_fragment()


# ──────────────────────────────────────────
with tab_upload:
    camera_photo = st.camera_input("Ambil foto")
    uploaded     = st.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])
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
                            f'<audio autoplay><source src="data:audio/wav;base64,{alarm_b64}"'
                            f' type="audio/wav"></audio>',
                            height=0,
                        )
                        st.subheader("💡 Saran")
                        for t in tips_for_label(label):
                            st.markdown(f"- {t}")
                    else:
                        st.success(f"✅ **{label}** ({conf:.2f})")
                    st.subheader(f"Top-{top_k}")
                    for i, c in topk_scores(scores, top_k):
                        st.metric(resolve_label(model.names, i), f"{c:.3f}")
