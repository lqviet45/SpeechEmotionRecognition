import streamlit as st
import librosa
import numpy as np
import joblib
import os
import io # Cần thiết để đọc file audio từ bộ nhớ
from audiorecorder import audiorecorder # Thư viện mới

# --- Cấu hình Trang (FE) ---
st.set_page_config(page_title="Nhận dạng Cảm xúc", layout="wide")
st.title("🎤 Ứng dụng Nhận dạng Cảm xúc Giọng nói (DSP501)")

# --- Tải Mô hình (BE) ---
# (Phần này giữ nguyên)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    model = joblib.load(os.path.join(MODEL_PATH, "rf_emotion_model_v1.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler_v1.pkl"))
    encoder = joblib.load(os.path.join(MODEL_PATH, "encoder_v1.pkl"))
    st.sidebar.success("Tải mô hình (RF), Scaler, và Encoder thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# --- Hàm Trích xuất Đặc trưng (BE) ---
# (Hàm này giữ nguyên)
def extract_features(y, sr=22050):
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        features = np.hstack((mfccs, rms, zcr))
        return features
    except Exception as e:
        st.error(f"Lỗi khi trích xuất đặc trưng: {e}")
        return None

# --- Hàm Xử lý và Dự đoán (BE) ---
# (Tách logic này ra hàm riêng để cả 2 tab cùng gọi)
def process_and_predict(y, sr):
    with st.spinner("Đang phân tích tín hiệu (DSP) và chạy mô hình AI..."):
        features = extract_features(y, sr)
        
        if features is not None:
            # Chuẩn bị dữ liệu (reshape và scale)
            features_2d = features.reshape(1, -1)
            features_scaled = scaler.transform(features_2d)
            
            # Dự đoán (AI/ML Model)
            prediction_encoded = model.predict(features_scaled)
            
            # Giải mã kết quả (Output)
            prediction_label = encoder.inverse_transform(prediction_encoded)[0]
            
            # Hiển thị kết quả (FE)
            st.subheader("Kết quả Phân loại:")
            st.success(f"Cảm xúc được dự đoán là: **{prediction_label.capitalize()}**")

# --- Giao diện (FE) dùng Tabs ---
tab1, tab2 = st.tabs(["📁 Tải file lên", "🎙️ Thu âm trực tiếp"])

# ----- Tab 1: Tải file lên -----
with tab1:
    st.header("Phương thức 1: Tải file âm thanh (.wav, .mp3)")
    uploaded_file = st.file_uploader("Chọn file âm thanh...", type=["wav", "mp3", "ogg"], key="file_uploader")

    if uploaded_file is not None:
        st.subheader("File âm thanh bạn đã tải lên:")
        st.audio(uploaded_file)
        
        # Tải file âm thanh bằng Librosa
        y, sr = librosa.load(uploaded_file, sr=22050)
        
        # Gọi hàm xử lý
        process_and_predict(y, sr)

# ----- Tab 2: Thu âm trực tiếp -----
with tab2:
    st.header("Phương thức 2: Thu âm giọng nói của bạn")
    st.write("Bấm nút bên dưới, nói, sau đó bấm dừng. App sẽ phân tích sau khi bạn bấm dừng.")

    # Widget thu âm
    audio_bytes = audiorecorder(
        start_prompt="Bấm để bắt đầu ghi âm ⏺️",
        stop_prompt="Bấm để dừng ghi âm ⏹️",
        pause_prompt="",
    )

    if audio_bytes:
        # Khi người dùng bấm dừng, audio_bytes sẽ có dữ liệu
        st.subheader("Bản thu âm của bạn:")
        st.audio(audio_bytes, format="audio/wav")

        # Chuyển audio_bytes (dữ liệu thô) thành một file-like object
        # mà Librosa có thể đọc được
        audio_file = io.BytesIO(audio_bytes)
        
        # Tải file âm thanh từ bộ nhớ
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Gọi hàm xử lý
        process_and_predict(y, sr)

# Thông tin sidebar
st.sidebar.info(
    "**Thông tin Dự án:**\n"
    "Môn học: DSP501\n"
    "Mô hình: Random Forest (Accuracy: 56.60%)\n"
    "Đặc trưng: MFCCs (20), Energy (1), ZCR (1)"
)