# 🚀 Dự án DSP501: Nhận dạng Cảm xúc Giọng nói

Đây là dự án cuối kỳ cho môn học DSP501, tích hợp các kỹ thuật Xử lý Tín hiệu Số (DSP) với Trí tuệ Nhân tạo (AI) để phân loại cảm xúc từ tín hiệu giọng nói.

## 📋 Quy trình Hệ thống (System Workflow)

Dự án tuân thủ theo quy trình 5 bước được yêu cầu trong đề bài:

**Signal Input → Preprocessing (DSP) → Feature Extraction → AI/ML Model → Classification Output**

## ✨ Tính năng

* Xử lý và trích xuất đặc trưng DSP (MFCCs, Energy, ZCR, Mel Spectrogram) từ file âm thanh.
* Huấn luyện và so sánh 3 mô hình AI/ML (SVM, Random Forest, CNN).
* Giải quyết vấn đề Overfitting của CNN bằng Early Stopping.
* Cung cấp một Web App Demo (Front-End) bằng Streamlit.
* Hỗ trợ dự đoán từ cả file tải lên và thu âm trực tiếp.

---

## ⚙️ Cài đặt và Yêu cầu Hệ thống

Để chạy dự án này, bạn cần cài đặt các thư viện Python VÀ các phần mềm bên ngoài (FFmpeg, CUDA).

### 1. Yêu cầu Bắt buộc (Bên ngoài)

Đây là các phần mềm phải được cài đặt trên hệ thống của bạn trước.

**a) Python (Quan trọng)**
* Dự án này **bắt buộc** phải sử dụng **Python 3.x.x**.

**b) FFmpeg (Bắt buộc cho Demo)**
* Thư viện thu âm (`streamlit-audiorecorder`) yêu cầu FFmpeg để xử lý file audio từ trình duyệt.
* **Cách cài:**
    1.  Tải bản "essentials build" từ: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
    2.  Giải nén và đặt thư mục `ffmpeg` vào `C:\ffmpeg`.
    3.  Thêm đường dẫn `C:\ffmpeg\bin` vào Biến môi trường (Environment Variables) `PATH` của hệ thống.
    4.  Khởi động lại Terminal và gõ `ffmpeg -version` để kiểm tra.

### 2. Yêu cầu Tùy chọn (Tăng tốc GPU)

Nếu bạn muốn huấn luyện mô hình CNN (Notebook 04) bằng GPU, bạn **bắt buộc** phải cài "công thức" sau:

* **GPU Driver:** NVIDIA Driver (phiên bản mới nhất).
* **TensorFlow:** `2.20.0` (đã có trong `requirements.txt`).
* **CUDA Toolkit:** **12.3** (Không phải 13.0).
* **cuDNN:** **8.9** (cho CUDA 12.x).
* Phải chạy ở WSL2 do từ tf 2.10 trở lên tf đã không hỗ trợ native windown

### 3. Cài đặt Môi trường Python

1.  **Clone dự án (Nếu có):**
    ```bash
    git clone [your-repo-link]
    cd DSP501_SpeechEmotionRecognition
    ```

2.  **Tạo môi trường ảo (Dùng Python 3.x):**
    ```bash
    py -3.x -m venv venv
    ```

3.  **Kích hoạt môi trường:**
    ```bash
    .\venv\Scripts\activate
    ```

4.  **Cài đặt các thư viện Python:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Hướng dẫn Chạy Dự án

Bạn có hai lựa chọn: (1) Chạy Demo để xem kết quả, hoặc (2) Huấn luyện lại từ đầu.

### 1. Cách chạy Demo (Nhanh nhất)

Cách này sử dụng các mô hình đã được huấn luyện (trong thư mục `models/`). Đảm bảo bạn đã cài **FFmpeg**.

1.  Kích hoạt môi trường ảo:
    ```bash
    .\venv\Scripts\activate
    ```
2.  Di chuyển vào thư mục `app`:
    ```bash
    cd app
    ```
3.  Chạy ứng dụng Streamlit:
    ```bash
    streamlit run app.py
    ```
4.  Trình duyệt sẽ tự động mở. Bạn có thể Tải file hoặc Thu âm.

### 2. Cách Huấn luyện lại Mô hình (Từ đầu)

Nếu bạn muốn tự mình chạy lại toàn bộ quy trình:

1.  **Tải Dữ liệu:**
    * Tải bộ dữ liệu **RAVDESS** (chỉ cần file `Audio_Speech_Actors_01-24.zip`).
    * Giải nén 24 thư mục (Actor_01...) vào thư mục `data/raw/`.

2.  **Chạy Notebooks (theo thứ tự):**
    * **(Khám phá)** `notebooks/01_data_exploration.ipynb`: Để xem dữ liệu và spectrogram.
    * **(Trích xuất cho SVM/RF)** `notebooks/02_feature_extraction.ipynb`: Chạy toàn bộ để tạo file `data/processed/features.npy` và `labels.npy`.
    * **(Huấn luyện SVM/RF)** `notebooks/03_model_training.ipynb`: Chạy toàn bộ để huấn luyện, đánh giá và lưu các file mô hình (`.pkl`) vào thư mục `models/`. **Demo sử dụng mô hình này.**
    * **(Trích xuất & Huấn luyện CNN)** `notebooks/04_cnn_model.ipynb`: Chạy toàn bộ để trích xuất spectrogram (X_cnn.npy) và huấn luyện mô hình CNN (với Early Stopping).
    * **(Kiểm tra Lọc)** `notebooks/05_filtering.ipynb`: Chạy để xác nhận yêu cầu lọc Bandpass.

---

## 📂 Cấu trúc Thư mục
```
DSP501_SpeechEmotionRecognition/
├── app/
│   └── app.py                    # Streamlit demo
├── data/
│   ├── raw/                      # Dữ liệu gốc RAVDESS (Actor_01...)
│   └── processed/                # File đã xử lý (.npy: features, labels, X_cnn…)
├── deliverables/                 # Báo cáo, slide, video nộp bài
├── models/
│   ├── rf_emotion_model_v1.pkl   # Mô hình Random Forest dùng cho Demo
│   └── (các file mô hình khác: .pkl, .h5, ...)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_cnn_model.ipynb
│   └── 05_filtering.ipynb
├── results/                      # Biểu đồ, hình ảnh và kết quả để chèn vào báo cáo
├── requirements.txt              # Danh sách thư viện Python
└── README.md                     # Hướng dẫn dự án
```