import streamlit as st
import librosa
import numpy as np
import joblib
import os
import io
from audiorecorder import audiorecorder
from scipy.signal import butter, lfilter

# --- Cấu hình Trang (FE) ---
st.set_page_config(page_title="Nhận dạng Cảm xúc", layout="wide")
st.title("🎤 Ứng dụng Nhận dạng Cảm xúc Giọng nói")

# --- HÀM LỌC (BE) ---
def butter_bandpass_filter(data, lowcut=100.0, highcut=8000.0, fs=22050, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, data)
    return y_filtered

# --- Tải Mô hình (BE) - V2 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    model = joblib.load(os.path.join(MODEL_PATH, "rf_emotion_model_v2.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler_v2.pkl"))
    encoder = joblib.load(os.path.join(MODEL_PATH, "encoder_v2.pkl"))
    st.sidebar.success("Tải mô hình (RF v2), Scaler (v2), và Encoder (v2) thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình v2: {e}")
    st.stop()

# --- Hàm Trích xuất Đặc trưng (BE) - (ĐÃ SỬA LỖI libS) ---
def extract_features(y, sr=22050):
    try:
        y_filtered = butter_bandpass_filter(y, fs=sr)
        mfccs = np.mean(librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=20).T, axis=0)
        
        rms = np.mean(librosa.feature.rms(y=y_filtered).T, axis=0) 
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_filtered).T, axis=0)
        features = np.hstack((mfccs, rms, zcr))
        return features
    except Exception as e:
        st.error(f"Lỗi khi trích xuất đặc trưng: {e}")
        return None

# --- Hàm Xử lý và Dự đoán (BE) ---
def process_and_predict(y, sr):
    with st.spinner("Đang áp dụng bộ lọc (DSP), trích xuất đặc trưng và chạy mô hình AI..."):
        features = extract_features(y, sr)
        
        if features is not None:
            features_2d = features.reshape(1, -1)
            features_scaled = scaler.transform(features_2d)
            
            prediction_encoded = model.predict(features_scaled)
            prediction_label = encoder.inverse_transform(prediction_encoded)[0]
            
            st.subheader("Kết quả Phân loại:")
            prediction_label_str = str(prediction_label).replace("np.str_('", "").replace("')", "")
            st.success(f"Cảm xúc được dự đoán là: **{prediction_label_str.capitalize()}**")

# --- Khởi tạo Session State (Quan trọng) ---
if 'last_processed_audio' not in st.session_state:
    st.session_state.last_processed_audio = None

# --- Giao diện (FE) dùng Tabs ---
tab1, tab2 = st.tabs(["📁 Tải file lên", "🎙️ Thu âm trực tiếp"])

# ----- Tab 1: Tải file lên -----
with tab1:
    st.header("Phương thức 1: Tải file âm thanh (.wav, .mp3)")
    uploaded_file = st.file_uploader("Chọn file âm thanh...", type=["wav", "mp3", "ogg"], key="file_uploader")

    if uploaded_file is not None:
        st.subheader("File âm thanh bạn đã tải lên:")
        st.audio(uploaded_file)
        
        y, sr = librosa.load(uploaded_file, sr=22050)
        process_and_predict(y, sr)

# ----- Tab 2: Thu âm trực tiếp (ĐÃ SỬA LỖI LOGIC) -----
with tab2:
    st.header("Phương thức 2: Thu âm giọng nói của bạn")
    st.write("Bấm nút bên dưới, nói, sau đó bấm dừng. App sẽ phân tích sau khi bạn bấm dừng.")

    audio_segment = audiorecorder(
        start_prompt="Bấm để bắt đầu ghi âm ⏺️",
        stop_prompt="Bấm để dừng ghi âm ⏹️",
        pause_prompt="",
    )

    if audio_segment:
        # KIỂM TRA XEM ĐÂY CÓ PHẢI BẢN THU ÂM MỚI KHÔNG
        if audio_segment != st.session_state.last_processed_audio:
            
            # 1. Đánh dấu là đã xử lý
            st.session_state.last_processed_audio = audio_segment
            
            st.subheader("Bản thu âm mới nhận được:")
            
            # 2. Chuyển đổi AudioSegment -> bytes
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()

            # 3. Phát âm thanh
            st.audio(wav_bytes, format="audio/wav")
            
            # 4. Tải vào Librosa
            audio_file_like = io.BytesIO(wav_bytes)
            y, sr = librosa.load(audio_file_like, sr=22050)
            
            # 5. Gọi hàm xử lý (CHỈ CHẠY 1 LẦN)
            process_and_predict(y, sr)
        
        # Nếu audio_segment giống hệt lần trước (do rerun),
        # code sẽ không chạy vào 'if' này và không dự đoán lại.
