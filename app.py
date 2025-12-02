import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(
    page_title="ANPR System - ITS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = 'models/best.pt'
DEFAULT_IMAGE = 'images/mobil.png'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di: {MODEL_PATH}. Silakan upload file best.pt ke folder 'models'.")
        return None
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_plate(image, upscale=3, gamma=1.5, thresh_val=150, 
                     crop_v=(10, 10), crop_h=(5, 5), thick=1, do_invert=True):
    """
    Fungsi untuk membersihkan gambar plat nomor sebelum dibaca OCR
    """
    h, w = image.shape[:2]

    if crop_v[0] + crop_v[1] < 100 and crop_h[0] + crop_h[1] < 100:
        y1 = int(h * (crop_v[0] / 100))
        y2 = h - int(h * (crop_v[1] / 100))
        x1 = int(w * (crop_h[0] / 100))
        x2 = w - int(w * (crop_h[1] / 100))
        image = image[y1:y2, x1:x2]

    h, w = image.shape[:2]
    if h == 0 or w == 0: return None
    image = cv2.resize(image, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)

    image = adjust_gamma(image, gamma=gamma)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    if do_invert:
        binary = cv2.bitwise_not(binary)

    if thick > 0:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=thick)
        
    return binary

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid_plate(text):
    pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$'
    return re.match(pattern, text) is not None


st.title("🚗 Sistem ANPR Indonesia")
st.markdown("Deteksi Plat Nomor menggunakan **YOLOv8** dan **EasyOCR**")

with st.sidebar:
    st.header("🎛️ Kendali Citra (Preprocessing)")
    
    st.subheader("1. Potong Sisi (Cropping)")
    crop_top = st.slider("Potong Atas (%)", 0, 30, 10)
    crop_bot = st.slider("Potong Bawah (%)", 0, 30, 10)
    crop_side = st.slider("Potong Samping (%)", 0, 20, 5)
    
    st.subheader("2. Binerisasi (Hitam Putih)")
    thresh_val = st.slider("Nilai Threshold", 0, 255, 140, help="Semakin kecil semakin hitam")
    invert_plate = st.checkbox("Balik Warna (Invert)", value=True, help="Centang jika hasil threshold tulisan putih background hitam")
    
    st.subheader("3. Kualitas Teks")
    gamma_val = st.slider("Gamma (Kecerahan)", 0.5, 3.0, 1.5)
    dilation = st.slider("Penebalan Huruf", 0, 3, 1)
    
    st.subheader("4. Model Config")
    conf_thresh = st.slider("Confidence YOLO", 0.1, 1.0, 0.25)

model = load_model()
reader = load_ocr()

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
elif os.path.exists(DEFAULT_IMAGE):
    image_pil = Image.open(DEFAULT_IMAGE)
else:
    image_pil = None

if image_pil is not None:
    image_pil = image_pil.convert("RGB") 
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image_pil, caption="Gambar Asli", use_container_width=True)
        run_btn = st.button("🔍 Deteksi Plat Nomor", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Sedang memproses..."):
            results = model.predict(image_cv, conf=conf_thresh)

            final_img = image_cv.copy()

            detected_plates = []

            for r in results:
                for box in r.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    plate_crop = image_cv[y1:y2, x1:x2]

                    processed_plate = preprocess_plate(
                        plate_crop, 
                        upscale=3,
                        gamma=gamma_val,
                        thresh_val=thresh_val,
                        crop_v=(crop_top, crop_bot),
                        crop_h=(crop_side, crop_side),
                        thick=dilation,
                        do_invert=invert_plate
                    )
                    
                    if processed_plate is not None:
                        ocr_result = reader.readtext(processed_plate, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                        plate_text = clean_text("".join(ocr_result))

                        is_valid = is_valid_plate(plate_text) and len(plate_text) > 2

                        box_color = (0, 255, 0) if is_valid else (0, 0, 255)

                        cv2.rectangle(final_img, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(final_img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        detected_plates.append({
                            "crop": plate_crop,
                            "processed": processed_plate,
                            "text": plate_text,
                            "score": score,
                            "valid": is_valid
                        })

            with col2:
                final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                st.image(final_rgb, caption="Hasil Deteksi", use_container_width=True)

            if len(detected_plates) > 0:
                st.success(f"Ditemukan {len(detected_plates)} plat nomor!")
                st.markdown("### Detail Pembacaan")
                
                for i, data in enumerate(detected_plates):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    
                    with c1:
                        st.image(cv2.cvtColor(data['crop'], cv2.COLOR_BGR2RGB), caption="Potongan Asli", width=150)
                    with c2:
                        st.image(data['processed'], caption="Hasil Preprocessing (Input OCR)", width=150)
                    with c3:
                        st.markdown(f"**Teks Terbaca:**")
                        if data['valid']:
                            st.markdown(f"## :green[{data['text']}]")
                        else:
                            st.markdown(f"## :red[{data['text']}]")
                            st.caption("Format tidak sesuai pola plat nomor / tidak terbaca jelas.")
                        
                        st.metric("YOLO Confidence", f"{data['score']*100:.1f}%")
            else:
                st.warning("YOLO tidak mendeteksi plat nomor pada gambar ini. Coba turunkan 'Confidence YOLO' di sidebar.")