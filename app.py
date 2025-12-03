import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
from PIL import Image
import os

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(
    page_title="ANPR System - Dual Model",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= KONFIGURASI MODEL =================
# Sesuai dengan screenshot folder kamu
MODEL_FRONT = 'models/front.pt'
MODEL_SIDE = 'models/side.pt'
DEFAULT_IMAGE = 'images/mobil.png'

@st.cache_resource
def load_models():
    """
    Load 2 Model sekaligus: Satu spesialis Depan, Satu spesialis Samping.
    """
    models = []
    
    # Load Model Front
    if os.path.exists(MODEL_FRONT):
        models.append(YOLO(MODEL_FRONT))
    else:
        st.error(f"⚠️ Model Depan tidak ditemukan: {MODEL_FRONT}")
        
    # Load Model Side
    if os.path.exists(MODEL_SIDE):
        models.append(YOLO(MODEL_SIDE))
    else:
        st.warning(f"⚠️ Model Samping tidak ditemukan: {MODEL_SIDE}. Hanya berjalan dengan 1 model.")
        
    return models

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ================= FUNGSI BANTUAN (HELPER) =================

def order_points(pts):
    # Urutkan titik koordinat untuk warping
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_warp(image):
    # Meluruskan plat nomor yang miring (Angle Samping)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    displayCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
            
    if displayCnt is None:
        return image

    rect = order_points(displayCnt.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def preprocess_plate(image, upscale=3, do_invert=True, do_warp=False):
    h, w = image.shape[:2]
    if h == 0 or w == 0: return None
    
    # 1. Perspective Warp (Opsional: Aktifkan di Sidebar buat foto miring)
    if do_warp:
        try:
            image = perspective_warp(image)
        except:
            pass 

    # 2. Upscaling
    h, w = image.shape[:2]
    image = cv2.resize(image, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)

    # 3. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Inversion (Wajib untuk Plat Hitam/Teks Putih)
    if do_invert:
        gray = cv2.bitwise_not(gray)

    # 5. Noise Removal (Haluskan dikit biar OCR gak pusing)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

def filter_text_by_size(ocr_results, height_thresh_ratio=0.5):
    """Filter teks kecil (tahun/bulan) berdasarkan rasio tinggi huruf utama"""
    if not ocr_results: return ""
    
    # Cari tinggi maksimum
    max_height = 0
    for (bbox, text, prob) in ocr_results:
        (tl, tr, br, bl) = bbox
        height = abs(br[1] - tr[1])
        if height > max_height: max_height = height
    
    # Ambil teks yang ukurannya cukup besar saja
    final_text_parts = []
    for (bbox, text, prob) in ocr_results:
        (tl, tr, br, bl) = bbox
        height = abs(br[1] - tr[1])
        if height >= max_height * height_thresh_ratio:
            final_text_parts.append(text)
            
    return "".join(final_text_parts)

def correct_common_mistakes(text):
    """Auto-correct kesalahan umum OCR (13 -> B, dll)"""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) < 2: return text
    
    # Logic: 13 di awal -> B
    if text.startswith("13"): text = "B" + text[2:]
    
    # Logic: Karakter pertama WAJIB Huruf
    first_char = text[0]
    corrections = {'1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '0': 'D'}
    if first_char.isdigit() and first_char in corrections:
        text = corrections[first_char] + text[1:]

    return text

def is_valid_plate(text):
    # Pola Regex Plat Indonesia
    pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$'
    return re.match(pattern, text) is not None

# ================= UI UTAMA =================

st.title("🚗 Sistem ANPR Indonesia (Dual Model)")
st.markdown("Menggunakan Ensemble Model: **Front View** + **Side View**")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎛️ Pengaturan Citra")
    
    st.subheader("1. Preprocessing Dasar")
    invert_plate = st.checkbox("Balik Warna (Invert)", value=True, help="Matikan ini jika Plat Putih (Tulisan Hitam).")
    warp_mode = st.checkbox("Koreksi Miring (Warp)", value=False, help="Aktifkan jika foto diambil dari samping ekstrim.")
    
    st.subheader("2. Padding (Anti-Potong)")
    st.info("Geser jika huruf pinggir (seperti 'B') tidak terbaca.")
    padding_x = st.slider("Padding Samping (px)", 0, 50, 15, help="Melebarkan kotak ke kiri & kanan")
    padding_y = st.slider("Padding Atas-Bawah (px)", 0, 50, 5)

    st.subheader("3. Konfigurasi AI")
    conf_thresh = st.slider("Confidence YOLO", 0.1, 1.0, 0.25)
    size_thresh = st.slider("Filter Teks Kecil (%)", 0.1, 1.0, 0.55, help="Hapus teks tahun/bulan expired")

# --- LOAD RESOURCES ---
loaded_models = load_models()
reader = load_ocr()

if not loaded_models:
    st.stop() # Stop jika tidak ada model sama sekali

uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "png", "jpeg"])

# --- MAIN LOGIC ---
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Gambar Asli", use_container_width=True)
        btn_run = st.button("🔍 Deteksi (2 Model)", type="primary", use_container_width=True)

    if btn_run:
        with st.spinner(f"Menjalankan {len(loaded_models)} Model YOLO secara bersamaan..."):
            
            final_img = image_cv.copy()
            detected_plates = []
            img_area = final_img.shape[0] * final_img.shape[1]
            h_img, w_img, _ = image_cv.shape

            # 1. JALANKAN SEMUA MODEL (ENSEMBLE)
            all_results = []
            for model in loaded_models:
                # Kumpulkan hasil dari front.pt DAN side.pt
                res = model.predict(image_cv, conf=conf_thresh)
                all_results.extend(res)

            # 2. PROSES SETIAP DETEKSI
            for r in all_results:
                for box in r.boxes.data.tolist():
                    x1, y1, x2, y2, score, cls = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # --- PERBAIKAN DI SINI ---
                    # Filter Fog Lamp (Objek terlalu kecil < 0.1% area gambar)
                    # Sebelumnya 0.005 (0.5%) terlalu besar, jadi plat jauh hilang.
                    if ((x2-x1)*(y2-y1)/img_area) < 0.001: continue 
                    # -------------------------

                    # --- LOGIKA PADDING (MELEBARKAN KOTAK) ---
                    # Menarik batas kotak agar huruf pinggir tidak terpotong
                    x1 = max(0, x1 - padding_x)
                    y1 = max(0, y1 - padding_y)
                    x2 = min(w_img, x2 + padding_x)
                    y2 = min(h_img, y2 + padding_y)
                    # -----------------------------------------

                    plate_crop = image_cv[y1:y2, x1:x2]
                    
                    # Preprocessing (Warping + Grayscale + Invert)
                    processed = preprocess_plate(plate_crop, upscale=3, do_invert=invert_plate, do_warp=warp_mode)
                    
                    if processed is not None:
                        # OCR Scan
                        ocr_raw = reader.readtext(processed, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        
                        # Filter (Hapus tahun) & Correct (Benerin Typo)
                        filtered_text = filter_text_by_size(ocr_raw, height_thresh_ratio=size_thresh)
                        final_text = correct_common_mistakes(filtered_text)
                        
                        # Validasi Format
                        is_valid = is_valid_plate(final_text) and len(final_text) > 2
                        color = (0, 255, 0) if is_valid else (0, 0, 255) # Hijau jika valid, Merah jika ragu

                        # Visualisasi
                        cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(final_img, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        detected_plates.append({
                            "crop": plate_crop, 
                            "processed": processed,
                            "text": final_text,
                            "score": score,
                            "valid": is_valid,
                            "model_origin": "YOLO" # Bisa dikembangkan nanti
                        })

            with col2:
                st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi Gabungan", use_container_width=True)
            
            # --- TABEL HASIL ---
            if detected_plates:
                st.success(f"Ditemukan {len(detected_plates)} kandidat plat nomor!")
                for i, data in enumerate(detected_plates):
                    with st.container():
                        c1, c2, c3 = st.columns([1, 1, 2])
                        with c1: st.image(data['crop'], caption=f"Crop #{i+1}")
                        with c2: st.image(data['processed'], caption="Input OCR")
                        with c3: 
                            st.markdown(f"### {data['text']}")
                            status = "✅ Valid" if data['valid'] else "⚠️ Format Tidak Pas"
                            st.caption(f"{status} | Conf: {data['score']:.2f}")
                            st.divider()
            else:
                st.warning("Tidak ditemukan plat nomor yang valid dari kedua model.")