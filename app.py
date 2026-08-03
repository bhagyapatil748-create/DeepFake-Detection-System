# ==========================================================
# IMPORTS
# ==========================================================

import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st
import time

from image_predict import predict_image
from video_predict import predict_video
from report import generate_report
from utils import get_file_type

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

/* Main background */
.stApp{
    background-color: var(--background-color);
}

/* Cards */
.result-card{
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border-radius:15px;
    padding:25px;
    border:1px solid rgba(128,128,128,0.2);
}

/* Headings */
.main-title,
.section-title{
    color: var(--text-color);
}

/* Text */
.processing,
.metric-label,
.confidence{
    color: var(--text-color);
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="DeepFake Image & Video Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

if "video_info" not in st.session_state:
    st.session_state.video_info = None


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image("logo.png", width=80)

    st.markdown("""
    # 🛡️ DeepFake Image & Video Detection

    Detect AI-generated **Images** and **Videos** using Deep Learning.

    Upload an image or video to receive:

    - ✅ Real / Fake Prediction
    - 📊 Confidence Score
    - 📄 PDF Report
    - 📚 Detection History
    - 📥 CSV Export

    ---
    """)
    st.markdown("---")

    st.subheader("Developer")

    st.write("Bhagya Patil")

    st.caption("AI & Python Developer")

    st.markdown("---")

    st.subheader("Supported Formats")

    st.write("🖼 JPG")
    st.write("🖼 JPEG")
    st.write("🖼 PNG")
    st.write("🎥 MP4")
    st.write("🎥 AVI")
    st.write("🎥 MOV")

    st.markdown("---")

    st.subheader("Model")

    st.success("EfficientNet-B0")

    st.caption("PyTorch")

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="main-title">
🛡️ DeepFake Image & Video Detection
</div>

<div class="subtitle">

AI Powered DeepFake Detection using
EfficientNet-B0 + LSTM

</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose an Image or Video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "mp4",
        "avi",
        "mov"
    ]
)


# ==========================================================
# AFTER UPLOAD
# ==========================================================

if uploaded_file is not None:

    file_type = get_file_type(uploaded_file.name)

    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.read())

    temp_file.close()

    temp_path = temp_file.name


    from PIL import Image

    # ======================================================
    # PREVIEW + FILE INFORMATION
    # ======================================================

    col1, col2 = st.columns([1, 2])

    # -------------------- Preview --------------------

    with col1:

        st.subheader(" Preview")

        if file_type == "image":

            # Open image
            preview = Image.open(temp_path)

            # Resize (Width, Height)
            preview = preview.resize((600, 600))

            # Display resized image
            st.image(preview)

        elif file_type == "video":

         st.video(temp_path)

        else:

            st.error("Unsupported File Type")
            st.stop()

    # -------------------- File Information --------------------

    with col2:

        st.subheader("📄 File Information")

        st.metric("Filename", uploaded_file.name)

        st.metric("Size", f"{uploaded_file.size/1024:.2f} KB")

        st.metric("Type", file_type.upper())

    # ======================================================
# ANALYZE BUTTON
# ======================================================

st.markdown("---")

if st.button(
    "🚀 Analyze Media",
    use_container_width=True
):

    with st.spinner("🧠 AI Model is analyzing..."):

        progress_text = st.empty()
        progress = st.progress(0)

        try:

            # -----------------------------------------
            # Progress Animation
            # -----------------------------------------

            for i in range(101):

                progress.progress(i)

                if i < 25:
                    progress_text.write("📂 Loading file...")
                elif i < 50:
                    progress_text.write("🖼️ Preprocessing...")
                elif i < 75:
                    progress_text.write("🧠 Running AI inference...")
                else:
                    progress_text.write("📊 Generating results...")

                time.sleep(0.01)

            # -----------------------------------------
            # Prediction
            # -----------------------------------------

            if file_type == "image":

                (
                    prediction,
                    confidence,
                    probabilities,
                    processing_time
                ) = predict_image(temp_path)

                video_info = None

            else:

                (
                    prediction,
                    confidence,
                    probabilities,
                    processing_time,
                    video_info
                ) = predict_video(temp_path)

            # -----------------------------------------
            # Store Results
            # -----------------------------------------

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.probabilities = probabilities
            st.session_state.processing_time = processing_time
            st.session_state.video_info = video_info

            # -----------------------------------------
            # Save History
            # -----------------------------------------

            st.session_state.history.append({

                "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

                "Filename": uploaded_file.name,

                "Type": file_type.title(),

                "Prediction": prediction,

                "Confidence": confidence

            })

        except Exception as e:

            st.error(f"❌ {e}")

        finally:

            progress.empty()
            progress_text.empty()

            if os.path.exists(temp_path):
                os.remove(temp_path)
# ==========================================================
# RESULTS DASHBOARD
# ==========================================================

if st.session_state.prediction is not None:

    st.markdown("---")
    st.header("📊 Prediction Result")

    left, right = st.columns([2, 1])

    # ======================================================
    # RESULT CARD
    # ======================================================

    with left:

        with st.container(border=True):

            if st.session_state.prediction == "Real":

                st.success("## ✅ REAL")

            else:

                st.error("## ❌ FAKE")

            st.metric(
                label="Confidence",
                value=f"{st.session_state.confidence:.2f}%"
            )

            st.caption(
                f"⏱ Processing Time : {st.session_state.processing_time:.3f} sec"
            )

    # ======================================================
    # PROBABILITIES
    # ======================================================

    with right:

        st.subheader("Probability")

        fake_prob = st.session_state.probabilities["Fake"]
        real_prob = st.session_state.probabilities["Real"]

        st.write("Fake")
        st.progress(fake_prob / 100)
        st.write(f"{fake_prob:.2f}%")

        st.write("Real")
        st.progress(real_prob / 100)
        st.write(f"{real_prob:.2f}%")

    # ======================================================
    # VIDEO INFORMATION
    # ======================================================

    if st.session_state.video_info is not None:

        st.markdown("---")

        st.subheader("🎥 Video Information")

        v1, v2, v3, v4 = st.columns(4)

        with v1:
            st.metric(
                "FPS",
                st.session_state.video_info["fps"]
            )

        with v2:
            st.metric(
                "Frames",
                st.session_state.video_info["frames"]
            )

        with v3:
            st.metric(
                "Duration",
                f'{st.session_state.video_info["duration"]:.2f}s'
            )

        with v4:
            st.metric(
                "Resolution",
                f'{st.session_state.video_info["width"]} × {st.session_state.video_info["height"]}'
            )

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.markdown("---")

    st.subheader("📄 Generate Report")

    report_folder = "reports"

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    report_name = (
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )

    report_path = os.path.join(
        report_folder,
        report_name
    )

    generate_report(

        output_path=report_path,

        filename=st.session_state.history[-1]["Filename"],

        file_type=st.session_state.history[-1]["Type"],

        prediction=st.session_state.prediction,

        confidence=st.session_state.confidence,

        probabilities=st.session_state.probabilities,

        processing_time=st.session_state.processing_time

    )

    with open(report_path, "rb") as pdf:

        st.download_button(

            label="⬇ Download PDF Report",

            data=pdf,

            file_name=report_name,

            mime="application/pdf",

            use_container_width=True

        )

# ==========================================================
# HISTORY
# ==========================================================

st.markdown("---")

st.header("📜 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "⬇ Download CSV",

        csv,

        "prediction_history.csv",

        "text/csv",

        use_container_width=True

    )

    if st.button(
        "🗑 Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.session_state.prediction = None

        st.session_state.confidence = None

        st.session_state.processing_time = None

        st.session_state.probabilities = None

        st.session_state.video_info = None

        st.rerun()

else:

    st.info("No predictions available.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center; color:gray;'>

### 🛡️ DeepFake Image & Video Detection System

Developed using

**Streamlit • PyTorch • EfficientNet-B0 • LSTM**

</div>
""",
unsafe_allow_html=True
)
