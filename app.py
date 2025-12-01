import os
os.environ['TMPDIR'] = '/tmp'

import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="ANPR System - Final Project", page_icon="🚗", layout="wide")
DEFAULT_IMAGE_PATH = "images/truck.png"

@st.cache_resource
def load_model():
    return YOLO('models/best.pt')

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

try:
    model = load_model()
    reader = load_ocr()
except:
    st.stop()

st.title("🚗 ANPR (Manual Control)")
st.markdown("---")

st.sidebar.header("1. Kendali Warna")
thresh_val = st.sidebar.slider("Manual Threshold", 0, 255, 170)
invert_plate = st.sidebar.checkbox("Balik Warna", value=True)

st.sidebar.header("2. Potong Bingkai")
crop_top = st.sidebar.slider("Potong Atas (%)", 0, 30, 15)
crop_bot = st.sidebar.slider("Potong Bawah (%)", 0, 40, 30)
crop_side = st.sidebar.slider("Potong Samping (%)", 0, 20, 5)

st.sidebar.header("3. Kualitas Huruf")
gamma_val = st.sidebar.slider("Gamma", 0.5, 3.5, 2.0)
dilation = st.sidebar.slider("Ketebalan Huruf", 0, 3, 1)

st.sidebar.header("4. Deteksi")
conf_thresh = st.sidebar.slider("Confidence YOLO", 0.1, 1.0, 0.25)
upscale_factor = st.sidebar.slider("Upscale", 2, 6, 4)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_plate(image, upscale=4, gamma=1.5, thresh=127, top=10, bot=10, side=5, thick=0, do_invert=False):
    h, w = image.shape[:2]
    if w < 10 or h < 10:
        return None

    c_top = int(h * (top / 100))
    c_bot = int(h * (bot / 100))
    c_side = int(w * (side / 100))

    if c_top + c_bot < h and 2 * c_side < w:
        image = image[c_top:h - c_bot, c_side:w - c_side]

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    image = cv2.resize(image, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    image = adjust_gamma(image, gamma=gamma)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    if do_invert:
        binary = cv2.bitwise_not(binary)

    if thick > 0:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=thick)

    binary = cv2.copyMakeBorder(binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return binary

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid_plate(text):
    pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$'
    return re.match(pattern, text) is not None

col1, col2 = st.columns(2)
image_source = None

uploaded = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
if uploaded:
    image_source = uploaded
elif os.path.exists(DEFAULT_IMAGE_PATH):
    image_source = DEFAULT_IMAGE_PATH

if image_source:
    image_pil = Image.open(image_source)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    with col1:
        st.image(image_pil, caption="Input", use_container_width=True)

        if st.button("🔍 Proses Deteksi", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                results = model.predict(image_cv, conf=conf_thresh)
                final_img = image_cv.copy()
                found_list = []

                st.sidebar.markdown("---")
                st.sidebar.subheader("Debug View")

                for r in results:
                    for box in r.boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        crop = image_cv[y1:y2, x1:x2]

                        processed = enhance_plate(
                            crop,
                            upscale=upscale_factor,
                            gamma=gamma_val,
                            thresh=thresh_val,
                            top=crop_top,
                            bot=crop_bot,
                            side=crop_side,
                            thick=dilation,
                            do_invert=invert_plate
                        )

                        if processed is not None:
                            raw = reader.readtext(
                                processed,
                                detail=0,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                            )

                            text = clean_text("".join(raw))

                            color = (0, 0, 255)
                            if is_valid_plate(text) and len(text) > 3:
                                color = (0, 255, 0)
                                found_list.append(text)
                            elif len(text) > 1:
                                color = (0, 255, 255)

                            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(final_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            st.sidebar.image(processed, caption=f"Baca: {text}", width=250)

                with col2:
                    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Hasil", use_container_width=True)

                    if found_list:
                        st.success(f"Valid: {found_list}")
                    else:
                        st.warning("Sesuaikan Manual Threshold untuk hasil terbaik.")
