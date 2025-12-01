# 🚗 Indonesian ANPR System (YOLOv8 + EasyOCR)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-yellow)

This project is a Final Project implementation of an **Automatic Number Plate Recognition (ANPR)** system specifically designed for Indonesian vehicle license plates. It addresses common challenges in real-world scenarios, such as low resolution, shadows, and uneven lighting.

The system integrates **YOLOv8** for robust object detection and **EasyOCR** for text recognition, wrapped in an interactive **Streamlit** web application. Users can manually tune preprocessing parameters (Threshold, Gamma, Cropping) in real-time to improve OCR accuracy on difficult images.

---

## 🌟 Key Features

* **Robust Detection:** Utilizes a custom-trained `YOLOv8n` model to accurately localize license plates.
* **Manual Preprocessing Control:** A unique feature allowing users to fine-tune image processing parameters when automatic OCR fails:
    * **Manual Thresholding:** Adjust black/white binarization limits.
    * **Gamma Correction:** Brighten dark/shadowed areas effectively.
    * **Smart Cropping:** Remove license plate frames and expiration dates that confuse the OCR.
    * **Auto Invert:** Automatically inverts colors (Black background to White) to match OCR requirements.
* **Debug View:** Visualize exactly what the OCR engine "sees" (preprocessed crop) for easier troubleshooting.

---

## 📖 How to Use (Web / Streamlit Cloud)

You can access the application via your browser here:
[Streamlit Cloud link](https://anpr-pcv.streamlit.app)

1.  **Upload Image:**
    * Upload a vehicle image (JPG/PNG).
    * If no image is uploaded, the system uses a default sample.

2.  **Click "Start Detection":**
    * The system will localize the plate and attempt to read the text.

3.  **Tuning (If the result is inaccurate):**
    Use the **Sidebar (Left Panel)** to adjust image parameters:
    * **Color Control:** Adjust the threshold until letters are **Bold Black** and background is **Clean White**.
    * **Crop Frame:** Use sliders to remove black borders around the plate.
    * **Gamma:** Increase gamma if the plate is covered in shadows.

4.  **Validation:**
    * **Green Box:** Valid license plate detected.
    * **Yellow/Red Box:** Plate detected but text format is invalid (Requires tuning).

---

## 💻 Local Installation Guide

Follow these steps to run the project on your local machine.

### Prerequisites
* Python 3.8 or higher.
* Git.

### Installation Steps (macOS / Linux)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/citylighxts/ANPR.git
    cd ANPR
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Environment:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

### Installation Steps (Windows)

1.  Open Command Prompt (CMD) or PowerShell in the project folder.
2.  Create environment: `python -m venv venv`
3.  Activate: `venv\Scripts\activate`
4.  Install: `pip install -r requirements.txt`
5.  Run: `streamlit run app.py`

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core logic. |
| **GUI** | Streamlit | Web framework for interactive data apps. |
| **Detection** | YOLOv8 (Ultralytics) | Object detection model for finding plates. |
| **OCR** | EasyOCR | Optical Character Recognition engine. |
| **Processing** | OpenCV | Image manipulation (Thresholding, Morphology, etc.). |

---

## 📂 Project Structure

```text
.
├── models/
│   └── best.pt           
├── images/
│   └── truck.png         
├── app.py                
├── requirements.txt      
└── README.md             