import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Object Detection and Tracking",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection and Tracking")
st.write("Upload a video to detect objects and track them using YOLO.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:
    if st.button("Start Detection and Tracking"):
        model = load_model()

        temp_input = None
        temp_output = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as input_file:
                input_file.write(uploaded_file.read())
                temp_input = input_file.name

            cap = cv2.VideoCapture(temp_input)

            if not cap.isOpened():
                st.error("Could not open this video. Please try another one.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if fps <= 0:
                    fps = 25

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as output_file:
                    temp_output = output_file.name

                writer = cv2.VideoWriter(
                    temp_output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)
                status = st.empty()

                frame_count = 0

                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    annotated_frame = results[0].plot()
                    writer.write(annotated_frame)

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.write(
                        f"Processing frame {frame_count}"
                        + (
                            f" of {total_frames}"
                            if total_frames > 0 else ""
                        )
                    )

                cap.release()
                writer.release()

                if frame_count > 0:
                    st.success("Detection and tracking completed!")

                    st.subheader("Processed Video")
                    st.video(temp_output)

                    with open(temp_output, "rb") as video_file:
                        st.download_button(
                            "Download Processed Video",
                            data=video_file.read(),
                            file_name="detected_tracked_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("No frames were found in this video.")

        except Exception as error:
            st.error(f"Something went wrong: {error}")

        finally:
            if "cap" in locals():
                cap.release()

            if "writer" in locals():
                writer.release()

            for path in (temp_input, temp_output):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

else:
    st.info("Upload a video to get started.")

st.divider()
st.caption("CodeAlpha AI Internship | Task 4")