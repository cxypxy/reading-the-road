import streamlit as st
import cv2
import numpy as np
from road_width import measure_road

st.set_page_config(page_title="Reading the Road", layout="wide")

st.title("Reading the Road")
st.caption("Upload a road photo. Get its estimated width in feet, with a confidence score.")

with st.sidebar:
    st.header("Settings")
    height_known = st.checkbox("I know the camera height", value=False)
    if height_known:
        cam_h = st.slider("Camera height (m)", 0.5, 3.0, 1.2, 0.1)
    else:
        cam_h = 1.5
        st.info("Using an assumed camera height of 1.5 m (4.9 ft). Results are approximate.")
    st.markdown("Tips for good results:")
    st.markdown("- Flat road, no steep hills")
    st.markdown("- Camera roughly level")
    st.markdown("- Road visible in lower half")
    st.markdown("- Daylight, clear view")
    st.markdown("- Not heavily occluded by vehicles")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("This file could not be decoded as an image. Try a JPG or PNG.")
        st.stop()

    with st.spinner("Measuring road geometry..."):
        result = measure_road(img_bgr, camera_height_m=cam_h)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Result")
        st.image(cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Road Width", f"{result.width_m:.2f} m", f"+/- {result.std_m:.2f} m")
    m2.metric("Confidence", f"{result.confidence * 100:.0f}%")
    m3.metric("Trust Level", result.trust)
    st.caption(f"Equivalent width: {result.width_m * 3.28084:.2f} ft +/- {result.std_m * 3.28084:.2f} ft")

    for warning in result.warnings or []:
        st.warning(warning)

    if not height_known:
        st.warning("Camera height was unknown, so this estimate uses 1.5 m. Verify before making critical decisions.")

    if result.trust == "Low":
        st.warning("Low confidence. Try a clearer photo.")
    elif result.trust == "Medium":
        st.info("Medium confidence. Verify for critical use.")
    else:
        st.success("High confidence. Measurement is reliable.")

    with st.expander("How this works"):
        st.markdown("1. GrabCut proposes the connected road surface from the lower image.")
        st.markdown("2. Scan lines recover left and right boundaries; sparse masks use a visible fallback.")
        st.markdown("3. Boundary intersection estimates the vanishing point and horizon.")
        st.markdown("4. Ground-plane pinhole geometry uses camera height and perspective depth.")
        st.markdown("5. Median scan-line width is reported, with spread and consistency informing confidence.")
        st.markdown("The result is an engineering estimate, not a substitute for a calibrated survey.")