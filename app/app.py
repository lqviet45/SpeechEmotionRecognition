import streamlit as st
import librosa
import numpy as np
import joblib
import os
import io
from audiorecorder import audiorecorder
from scipy.signal import butter, lfilter # <-- THÊM MỚI ĐỂ LỌC

# --- Cấu hình Trang (FE) ---
st.set_page_config(page_title="Nhận dạng Cảm xúc", layout="wide")
st.title("🎤 Ứng dụng Nhận dạng Cảm xúc Giọng nói (DSP501) - V2 (Đã lọc)")

# --- HÀM LỌC (THÊM MỚI) ---
def butter_bandpass_filter(data, lowcut=100.0, highcut=8000.0, fs=22050, order=5):
    """
    Hàm thiết kế và áp dụng bộ lọc bandpass Butterworth.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, data)
    return y_filtered

# --- Tải Mô hình (BE) - ĐÃ CẬP NHẬT LÊN V2 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    # Tải các file v2 mới
    model = joblib.load(os.path.join(MODEL_PATH, "rf_emotion_model_v2.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler_v2.pkl"))
    encoder = joblib.load(os.path.join(MODEL_PATH, "encoder_v2.pkl"))
    
    st.sidebar.success("Tải mô hình (RF v2), Scaler (v2), và Encoder (v2) thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình v2: {e}")
    st.stop()

# --- Hàm Trích xuất Đặc trưng (BE) - ĐÃ CẬP NHẬT ĐỂ LỌC ---
def extract_features(y, sr=22050):
    try:
        # 1. ÁP DỤNG BỘ LỌC (BƯỚC MỚI)
        y_filtered = butter_bandpass_filter(y, fs=sr)
        
        # 2. Trích xuất đặc trưng từ tín hiệu ĐÃ LỌC (y_filtered)
        mfccs = np.mean(librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=20).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y_filtered).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_filtered).T, axis=0)
        
        # 3. Kết hợp
        features = np.hstack((mfccs, rms, zcr))
        return features
    except Exception as e:
        st.error(f"Lỗi khi trích xuất đặc trưng: {e}")
        return None

# --- Hàm Xử lý và Dự đoán (BE) ---
# (Hàm này giữ nguyên)
def process_and_predict(y, sr):
    with st.spinner("Đang áp dụng bộ lọc (DSP), trích xuất đặc trưng và chạy mô hình AI..."):
        features = extract_features(y, sr)
        
        if features is not None:
            features_2d = features.reshape(1, -1)
            features_scaled = scaler.transform(features_2d)
            
            prediction_encoded = model.predict(features_scaled)
            prediction_label = encoder.inverse_transform(prediction_encoded)[0]
            
            st.subheader("Kết quả Phân loại:")
            # Sửa lại tên class (loại bỏ np.str_)
            prediction_label_str = str(prediction_label).replace("np.str_('", "").replace("')", "")
            st.success(f"Cảm xúc được dự đoán là: **{prediction_label_str.capitalize()}**")

# --- Giao diện (FE) dùng Tabs ---
# (Phần này giữ nguyên)
tab1, tab2 = st.tabs(["📁 Tải file lên", "🎙️ Thu âm trực tiếp"])

with tab1:
    st.header("Phương thức 1: Tải file âm thanh (.wav, .mp3)")
    uploaded_file = st.file_uploader("Chọn file âm thanh...", type=["wav", "mp3", "ogg"], key="file_uploader")

    if uploaded_file is not None:
        st.subheader("File âm thanh bạn đã tải lên:")
        st.audio(uploaded_file)
        
        y, sr = librosa.load(uploaded_file, sr=22050)
        process_and_predict(y, sr)

with tab2:
    st.header("Phương thức 2: Thu âm giọng nói của bạn")
    st.write("Bấm nút bên dưới, nói, sau đó bấm dừng. App sẽ phân tích sau khi bạn bấm dừng.")

    audio_bytes = audiorecorder(
        start_prompt="Bấm để bắt đầu ghi âm ⏺️",
        stop_prompt="Bấm để dừng ghi âm ⏹️",
        pause_prompt="",
    )

    if audio_bytes:
        st.subheader("Bản thu âm của bạn:")
        st.audio(audio_bytes, format="audio/wav")
        audio_file = io.BytesIO(audio_bytes)
        
        y, sr = librosa.load(audio_file, sr=22050)
        process_and_predict(y, sr)

# --- Sidebar (ĐÃ CẬP NHẬT) ---
st.sidebar.info(
    "**Thông tin Dự án:**\n"
    "Môn học: DSP501\n"
    "Mô hình: Random Forest (v2 - Đã lọc)\n"
    "Độ chính xác: **57.99%**\n"
    "Đặc trưng: Filtered MFCCs (20), Energy (1), ZCR (1)"
)