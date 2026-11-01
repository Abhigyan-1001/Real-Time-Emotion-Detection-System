"""Real-Time Emotion Detection System: Streamlit dashboard.

Run with:  python -m streamlit run app.py
"""

import time
from pathlib import Path

import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.face_detector import FaceDetector
from src.predictor import EmotionPredictor
from src.session import LiveAnalyzer, SessionSnapshot

ROOT = Path(__file__).resolve().parent
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_PATH = ROOT / "models" / "emotion_cnn_v1.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
UNCERTAIN = "uncertain"

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="🙂", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Runs once per server process, not on every rerun."""
    return FaceDetector(YUNET_PATH), EmotionPredictor(MODEL_PATH, LABELS_PATH)


try:
    detector, predictor = load_models()
except Exception as exc:  # show a readable message instead of a traceback
    st.error(f"Could not load the models: {exc}")
    st.info(
        "Expected these files in the `models/` folder: "
        "`face_detection_yunet_2023mar.onnx`, `emotion_cnn_v1.keras`, `labels.json`."
    )
    st.stop()

# One analyzer per browser session; session_state survives Streamlit reruns.
if "analyzer" not in st.session_state:
    st.session_state.analyzer = LiveAnalyzer(detector, predictor)
analyzer: LiveAnalyzer = st.session_state.analyzer

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    alpha = st.slider(
        "Smoothing strength (alpha)",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Smoothed = alpha * current + (1 - alpha) * previous. "
        "1.0 = no smoothing. Smaller = steadier but slower to react.",
    )
    threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.0,
        max_value=0.9,
        value=0.40,
        step=0.05,
        help="Below this top score the app shows 'uncertain'. "
        "The default (0.40) was chosen using validation results and a live smoothing test; see the README.",
    )
    if st.button("Reset session statistics"):
        analyzer.reset()
    st.caption(
        "Privacy: video is processed in memory on this machine. "
        "Nothing is recorded or saved; only counts and timings are kept."
    )
    with st.expander("About this model"):
        st.write(
            "A custom CNN trained on FER-2013 (48x48 grayscale faces, 7 classes). "
            "On the validation split it reached about 61.6% accuracy and 0.585 macro-F1 "
            "(single run). The held-out test set has not been evaluated yet. "
            "Fear, sad and angry are the weakest classes. "
            "Live webcam images differ from the training data, so accuracy may be lower."
        )

analyzer.configure(alpha=alpha, threshold=threshold)

# ---------------------------------------------------------------- header
st.title("Real-Time Emotion Detection System")
st.caption(
    "Classifies the visible facial expression pattern using a model trained on labeled images. "
    "It cannot determine anyone's true internal emotional state."
)

col_video, col_stats = st.columns([3, 2])


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Runs in a worker thread: no st.* calls allowed here.
    img = frame.to_ndarray(format="bgr24")
    out = analyzer.process(img)
    return av.VideoFrame.from_ndarray(out, format="bgr24")


with col_video:
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False,
        },
    )

# Start a fresh session each time the stream goes from stopped to playing.
playing = bool(ctx.state.playing)
if playing and not st.session_state.get("was_playing", False):
    analyzer.reset()
st.session_state["was_playing"] = playing

stats_area = col_stats.empty()


def fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def render(snap: SessionSnapshot, is_playing: bool) -> None:
    with stats_area.container():
        # --- status messages
        if snap.last_error:
            st.error(f"Processing error (the stream keeps running): {snap.last_error}")
        elif snap.frames_total == 0:
            st.info(
                "Waiting for the first frame..."
                if is_playing
                else "Press START under the video and allow camera access."
            )
        elif is_playing and snap.faces_in_frame == 0:
            st.warning("No face detected. Face the camera in good lighting.")
        elif is_playing and snap.current_uncertain:
            st.info("The model's top score is below the threshold, so no label is shown.")
        elif not is_playing:
            st.caption("Stream stopped. The last session's results stay visible until you press START again.")

        # --- current prediction
        c1, c2 = st.columns(2)
        c1.metric("Current prediction", snap.current_label or "-")
        c2.metric(
            "Model score",
            f"{snap.current_score:.0%}" if snap.current_score is not None else "-",
            help="The model's softmax output for the top class. It is not a verified probability of being correct.",
        )

        # --- session metrics
        c3, c4, c5 = st.columns(3)
        c3.metric("Session time", fmt_duration(snap.duration_s))
        c4.metric("Processing FPS", f"{snap.fps:.1f}")
        c5.metric(
            "Callback latency",
            f"{snap.latency_ms_mean:.0f} ms",
            help=f"Mean time inside our frame callback (detect + classify + track + draw). "
            f"95th percentile: {snap.latency_ms_p95:.0f} ms. Excludes camera and network time.",
        )

        # --- prediction distribution
        st.subheader("Predictions this session")
        order = predictor.labels + [UNCERTAIN]
        counts = [snap.prediction_counts.get(name, 0) for name in order]
        if sum(counts) > 0:
            st.bar_chart(pd.DataFrame({"count": counts}, index=order))
        st.caption(
            "Counts what the model predicted per face per processed frame. "
            "These are model outputs, not measurements of anyone's true emotions."
        )


render(analyzer.snapshot(), playing)
while ctx.state.playing:
    time.sleep(0.5)
    render(analyzer.snapshot(), True)