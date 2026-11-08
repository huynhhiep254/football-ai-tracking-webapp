
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

VIDEO_PATH = ROOT / "outputs" / "final_demo" / "match_roboflow_web.mp4"

CSV_PATH = ROOT / "outputs" / "final_demo" / "match_roboflow_final.csv"

st.set_page_config(
    page_title="Football AI Tracking",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football AI Tracking")
st.caption("Football Player Detection & Tracking using YOLOv8 + ByteTrack")

st.markdown(
    """
    This application demonstrates an AI system for detecting
    and tracking football players in match videos.

    **Object classes:** Player, Goalkeeper, Referee, Ball.
    """
)

st.divider()

st.subheader("🎬 Tracking Video")

if VIDEO_PATH.exists():
    st.video(str(VIDEO_PATH))
else:
    st.warning("Demo video not found.")


# ======================================================
# UPLOAD AND RUN YOLOv8 + BYTETRACK
# ======================================================

st.subheader("📤 Upload & Analyze Football Video")

uploaded_video = st.file_uploader(
    "Choose an MP4 football video",
    type=["mp4"],
    key="football_upload"
)

if uploaded_video is not None:

    size_mb = uploaded_video.size / (1024 * 1024)

    st.success(f"Uploaded: {uploaded_video.name}")
    st.write(f"Video size: {size_mb:.2f} MB")

    if size_mb > 200:
    st.warning("Please choose a video under 200 MB.")
    else:
        st.video(uploaded_video.getvalue())

        if st.button(
            "🚀 Run YOLOv8 + ByteTrack",
            type="primary"
        ):
            import sys
            import subprocess
            import uuid
            import imageio_ffmpeg

            upload_dir = ROOT / "videos" / "web_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            unique_name = "web_" + uuid.uuid4().hex[:10]

            input_path = upload_dir / f"{unique_name}.mp4"
            input_path.write_bytes(uploaded_video.getvalue())

            output_dir = ROOT / "outputs" / "final_demo"

            raw_output = output_dir / f"{unique_name}_preview.mp4"
            csv_output = output_dir / f"{unique_name}_preview.csv"
            web_output = output_dir / f"{unique_name}_preview_web.mp4"

            command = [
                sys.executable,
                str(ROOT / "src" / "09_final_demo.py"),
                "--input",
                str(input_path),
                "--seconds",
                "2"
            ]

            try:
                with st.spinner("Running YOLOv8 + ByteTrack..."):
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=600
                    )

                if result.returncode != 0:
                    st.error("AI processing failed.")
                    st.code((result.stdout + result.stderr)[-5000:])

                elif not raw_output.exists() or not csv_output.exists():
                    st.error("AI finished but output files were not found.")

                else:
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

                    convert_command = [
                        ffmpeg,
                        "-n",
                        "-i",
                        str(raw_output),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-movflags",
                        "+faststart",
                        "-an",
                        str(web_output)
                    ]

                    with st.spinner("Preparing browser-compatible video..."):
                        subprocess.run(
                            convert_command,
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )

                    st.session_state["ai_results"] = {
                        "video": str(web_output),
                        "csv": str(csv_output)
                    }

                    st.success("AI processing completed!")

            except subprocess.TimeoutExpired:
                st.error("Processing exceeded the time limit.")

            except Exception as error:
                st.error(f"Processing error: {error}")

        # Keep results visible after Streamlit reruns
        results = st.session_state.get("ai_results")

        if results is not None:
            result_video = Path(results["video"])
            result_csv = Path(results["csv"])

            if result_video.is_file() and result_csv.is_file():

                st.subheader("🎬 AI Tracking Result")
                st.video(str(result_video))

                st.subheader("📊 Tracking CSV")

                result_df = pd.read_csv(result_csv)
                st.dataframe(result_df.head(100))

                st.download_button(
                    "📥 Download Tracking CSV",
                    data=result_csv.read_bytes(),
                    file_name="tracking_result.csv",
                    mime="text/csv"
                )

                st.download_button(
                    "📥 Download Tracking Video",
                    data=result_video.read_bytes(),
                    file_name="tracking_result.mp4",
                    mime="video/mp4"
                )


st.divider()

st.subheader("📊 Detection & Tracking Statistics")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    player_df = df[df["class_name"] == "player"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Records", f"{len(df):,}")
    col2.metric("Player Track IDs", player_df["track_id"].nunique())
    col3.metric("Object Classes", df["class_name"].nunique())

    st.subheader("Detections by Class")
    st.bar_chart(df["class_name"].value_counts())

    st.subheader("Tracking Data")
    st.dataframe(df.head(100), use_container_width=True)

    st.download_button(
        label="📥 Download Tracking CSV",
        data=CSV_PATH.read_bytes(),
        file_name="football_tracking.csv",
        mime="text/csv",
    )
else:
    st.warning("Tracking CSV not found.")

st.divider()
st.caption("Academic Project | YOLOv8 + ByteTrack")

