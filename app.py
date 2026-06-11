from __future__ import annotations

import base64
import io
from pathlib import Path
import threading
import time
import wave
from typing import Iterable

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_IMG_SIZE = 224
DEFAULT_ALARM_THRESHOLD = 0.7
ALARM_COOLDOWN_SEC = 2.0
ALARM_POLL_INTERVAL_SEC = 0.2
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


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_label(names: object, index: int) -> str:
    if isinstance(names, dict):
        return names.get(index, str(index))
    if isinstance(names, (list, tuple)) and index < len(names):
        return str(names[index])
    return str(index)


def topk_scores(scores: np.ndarray, k: int) -> Iterable[tuple[int, float]]:
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
    bad_labels = [label for label in labels if any(hint in label.lower() for hint in DEFAULT_BAD_LABEL_HINTS)]
    if bad_labels:
        return bad_labels
    return [
        label
        for label in labels
        if not any(hint in label.lower() for hint in DEFAULT_GOOD_LABEL_HINTS)
    ]


def tips_for_label(label: str) -> list[str]:
    lower_label = label.lower()
    for key, tips in TIP_RULES.items():
        if key in lower_label:
            return list(tips)
    return list(DEFAULT_TIPS)


@st.cache_data
def alarm_audio_data_uri() -> str:
    duration = 0.25
    sample_rate = 22050
    frequency = 880.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (tone * np.iinfo(np.int16).max).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(audio.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def play_alarm_audio() -> None:
    audio_uri = alarm_audio_data_uri()
    st.components.v1.html(
        f'<audio autoplay="true"><source src="{audio_uri}" type="audio/wav"></audio>',
        height=0,
    )


class PostureVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        model: YOLO,
        names: object,
        conf_threshold: float,
        top_k: int,
        img_size: int,
        bad_labels: Iterable[str],
        alarm_threshold: float,
    ) -> None:
        self.model = model
        self.names = names
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.img_size = img_size
        self.bad_labels = {str(label) for label in bad_labels}
        self.alarm_threshold = alarm_threshold
        self.lock = threading.Lock()
        self.last_pred: tuple[str, float] | None = None
        self.last_is_bad = False

    def update_settings(
        self,
        conf_threshold: float,
        top_k: int,
        img_size: int,
        alarm_threshold: float,
        bad_labels: Iterable[str],
    ) -> None:
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.img_size = img_size
        self.alarm_threshold = alarm_threshold
        self.bad_labels = {str(label) for label in bad_labels}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        results = self.model.predict(image, imgsz=self.img_size, verbose=False)
        if results:
            probs = getattr(results[0], "probs", None)
            if probs is not None and getattr(probs, "data", None) is not None:
                scores = probs.data
                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                else:
                    scores = np.asarray(scores)
                scores = np.asarray(scores).reshape(-1)

                if scores.size:
                    top1_index = int(scores.argmax())
                    top1_label = resolve_label(self.names, top1_index)
                    top1_conf = float(scores[top1_index])
                    is_bad = top1_label in self.bad_labels and top1_conf >= self.alarm_threshold
                    with self.lock:
                        self.last_pred = (top1_label, top1_conf)
                        self.last_is_bad = is_bad

                    if is_bad:
                        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
                        cv2.putText(
                            image,
                            "WARNING: BAD POSTURE",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                        )
                else:
                    with self.lock:
                        self.last_pred = None
                        self.last_is_bad = False

                line_height = 28
                num_lines = min(self.top_k, int(scores.size)) if scores.size else 1
                y = max(line_height, height - 10 - line_height * (num_lines - 1))
                for index, conf in topk_scores(scores, self.top_k):
                    label = resolve_label(self.names, index)
                    color = (0, 255, 0) if conf >= self.conf_threshold else (0, 165, 255)
                    text = f"{label}: {conf:.2f}"
                    cv2.putText(
                        image,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    y += 28

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Posture Classifier", layout="wide")

st.title("Posture classifier")
st.write("Allow camera access when prompted to run live predictions.")

weights = sorted([p.name for p in WEIGHTS_DIR.glob("*.pt")])
if not weights:
    st.error(f"No .pt model weights found in {WEIGHTS_DIR}")
    st.stop()

selected_weight = st.sidebar.selectbox(
    "Model weights",
    weights,
    index=weights.index("best.pt") if "best.pt" in weights else 0,
)
img_size = st.sidebar.select_slider(
    "Image size",
    options=[160, 192, 224, 256, 288, 320],
    value=DEFAULT_IMG_SIZE,
)
conf_threshold = st.sidebar.slider("Display threshold", 0.0, 1.0, 0.5, 0.01)
top_k = st.sidebar.slider("Top-K", 1, 5, 2)
alarm_threshold = st.sidebar.slider(
    "Alarm threshold",
    0.0,
    1.0,
    DEFAULT_ALARM_THRESHOLD,
    0.01,
)

model_path = WEIGHTS_DIR / selected_weight
model = load_model(str(model_path))
label_options = list_labels(model.names)
bad_labels = infer_bad_labels(label_options)
if bad_labels:
    st.sidebar.caption("Bad posture labels: " + ", ".join(bad_labels))
else:
    st.sidebar.caption("Bad posture labels: none detected from names.")

st.caption("Live camera")

webrtc_ctx = webrtc_streamer(
    key="posture-classifier",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: PostureVideoProcessor(
        model,
        model.names,
        conf_threshold,
        top_k,
        img_size,
        bad_labels,
        alarm_threshold,
    ),
    async_processing=True,
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.update_settings(
        conf_threshold,
        top_k,
        img_size,
        alarm_threshold,
        bad_labels,
    )

alarm_box = st.empty()
tips_box = st.empty()

st.divider()
st.caption("Single image fallback")

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not read the image file.")
    else:
        results = model.predict(image, imgsz=img_size, verbose=False)
        probs = getattr(results[0], "probs", None)
        if probs is None or getattr(probs, "data", None) is None:
            st.error("No classification scores returned.")
        else:
            scores = probs.data
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            else:
                scores = np.asarray(scores)
            scores = np.asarray(scores).reshape(-1)

            if scores.size:
                top1_index = int(scores.argmax())
                top1_label = resolve_label(model.names, top1_index)
                top1_conf = float(scores[top1_index])
                is_bad = top1_label in bad_labels and top1_conf >= alarm_threshold
                if is_bad:
                    st.error(f"ALARM: postur buruk terdeteksi ({top1_label}, {top1_conf:.2f})")
                    tips = tips_for_label(top1_label)
                    st.info("Saran:\n" + "\n".join(f"- {tip}" for tip in tips))

            lines = []
            for index, conf in topk_scores(scores, top_k):
                label = resolve_label(model.names, index)
                lines.append(f"{label}: {conf:.3f}")
            st.write("\n".join(lines))
            st.image(image, channels="BGR", caption="Input")

if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    while webrtc_ctx.state.playing:
        with webrtc_ctx.video_processor.lock:
            last_pred = webrtc_ctx.video_processor.last_pred
            last_is_bad = webrtc_ctx.video_processor.last_is_bad
        if last_pred is None or not last_is_bad:
            alarm_box.empty()
            tips_box.empty()
        else:
            label, conf = last_pred
            alarm_box.error(f"ALARM: postur buruk terdeteksi ({label}, {conf:.2f})")
            tips = tips_for_label(label)
            tips_box.info("Saran:\n" + "\n".join(f"- {tip}" for tip in tips))
            last_alarm_ts = st.session_state.get("last_alarm_ts", 0.0)
            now = time.time()
            if now - last_alarm_ts >= ALARM_COOLDOWN_SEC:
                play_alarm_audio()
                st.session_state["last_alarm_ts"] = now
        time.sleep(ALARM_POLL_INTERVAL_SEC)
