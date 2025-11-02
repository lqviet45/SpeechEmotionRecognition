# --- [PHẦN A] IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os
import glob
import librosa  # Dùng để phân tích, xử lý tín hiệu âm thanh (Tải file, tính MFCCs,...)
import numpy as np  # Dùng để xử lý mảng (array)
from tqdm import tqdm  # Dùng để tạo thanh tiến trình (progress bar)
from scipy.signal import butter, lfilter  # Dùng để thiết kế và áp dụng bộ lọc DSP (IIR)
import joblib  # Dùng để lưu và tải mô hình AI (hiệu quả hơn 'pickle' cho mảng numpy)

# Import các thư viện AI/ML từ Scikit-learn
from sklearn.model_selection import train_test_split  # Dùng để chia dữ liệu thành tập train/test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Dùng để chuẩn hóa dữ liệu và mã hóa nhãn
from sklearn.metrics import accuracy_score, classification_report  # Dùng để đánh giá mô hình
from sklearn.ensemble import RandomForestClassifier  # Mô hình AI/ML (Rừng Ngẫu nhiên)

# --- [PHẦN B] CẤU HÌNH VÀ CÁC HẰNG SỐ ---

# Định nghĩa các đường dẫn thư mục.
# (Giả định rằng bạn chạy file này từ thư mục gốc 'DSP501_SpeechEmotionRecognition/')
RAW_DATA_DIR = "data/raw/"  # Nơi chứa 1440 file .wav gốc
PROCESSED_DATA_DIR = "data/processed/"  # Nơi lưu dữ liệu .npy (kết quả của Bước 1)
MODEL_DIR = "models/"  # Nơi lưu các mô hình .pkl đã huấn luyện (kết quả của Bước 2)

# Đảm bảo các thư mục này tồn tại trước khi lưu file vào
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Ánh xạ (map) nhãn từ tên file
# Tên file RAVDESS (ví dụ: 03-01-03-01-01-01-01.wav) có số '03' ở vị trí thứ 3
# 'emotion_map' giúp dịch số '03' này thành nhãn chữ 'happy'
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# --- [PHẦN C] CÁC HÀM XỬ LÝ (DSP VÀ AI/ML) ---

# === BƯỚC 1: HÀM DSP (Preprocessing & Feature Extraction) ===

def butter_bandpass_filter(data, lowcut=100.0, highcut=8000.0, fs=22050, order=5):
    """
    Hàm thiết kế và áp dụng bộ lọc Bandpass (Thông dải).
    Đây là yêu cầu 'Digital filtering' của đề bài.
    - Tại sao dùng? Để loại bỏ nhiễu tần số rất thấp (như tiếng ồn 50Hz)
      và nhiễu tần số rất cao (tiếng rít) mà giọng nói không chứa.
    - lowcut=100Hz: Cắt bỏ âm thanh dưới 100Hz (tiếng ồn nền).
    - highcut=8000Hz (8kHz): Cắt bỏ âm thanh trên 8kHz (tiếng rít).
    - fs=22050: Tần số lấy mẫu (sample rate) của tín hiệu.
    - order=5: Bậc của bộ lọc. Bậc càng cao, bộ lọc càng "dốc" (cắt gắt hơn).
    """
    # Tần số Nyquist = 1/2 tần số lấy mẫu (theo lý thuyết)
    nyq = 0.5 * fs
    
    # Chuẩn hóa tần số cắt (0.0 -> 1.0)
    low = lowcut / nyq
    high = highcut / nyq
    
    # Tính toán các hệ số (b, a) của bộ lọc IIR (Butterworth)
    b, a = butter(order, [low, high], btype='band')
    
    # Áp dụng bộ lọc lên tín hiệu 'data'
    y_filtered = lfilter(b, a, data)
    return y_filtered

def extract_features_filtered(file_path):
    """
    Hàm trích xuất đặc trưng V2 (đã lọc) cho MỘT file âm thanh.
    Đây là cốt lõi của quy trình 'Preprocessing' và 'Feature Extraction'.
    """
    try:
        # 1. Input & Preprocessing (Sampling):
        # Tải file âm thanh. 'sr=22050' thực hiện việc TÁI LẤY MẪU (resampling).
        # - Tại sao 22050Hz? Giọng nói hiếm khi vượt 10kHz, 22050Hz là đủ
        #   (theo lý thuyết Nyquist) và giúp giảm 50% khối lượng tính toán so với 44100Hz.
        y, sr = librosa.load(file_path, sr=22050)
        
        # 2. Preprocessing (Filtering):
        # Áp dụng bộ lọc bandpass chúng ta vừa định nghĩa.
        y_filtered = butter_bandpass_filter(y, fs=sr)
        
        # 3. Feature Extraction (Trích xuất Đặc trưng):
        # Chúng ta trích xuất 3 loại đặc trưng từ tín hiệu ĐÃ LỌC (y_filtered)
        
        # Đặc trưng 1: MFCCs (20 giá trị)
        # - Tại sao dùng? MFCCs là đặc trưng 'vàng' trong nhận dạng giọng nói,
        #   nó mô phỏng cách tai người nghe và bắt được 'âm sắc' (timbre) của giọng.
        # - n_mfcc=20: Lấy 20 hệ số (là giá trị phổ biến).
        # - np.mean(..., .T, axis=0):
        #   - MFCCs trả về một mảng (20, N) với N là số khung thời gian.
        #   - Các file âm thanh có độ dài (N) khác nhau.
        #   - Ta không thể đưa dữ liệu (N) thay đổi vào SVM/RF.
        #   - Giải pháp: Lấy 'np.mean' (trung bình) theo trục thời gian (axis=0)
        #     để 'nén' mảng (20, N) thành một vector (20,) cố định.
        mfccs = np.mean(librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=20).T, axis=0)
        
        # Đặc trưng 2: Energy (Năng lượng) (1 giá trị)
        # - Tại sao dùng? Năng lượng (độ to) của giọng nói là đặc trưng quan trọng
        #   (ví dụ: 'angry' (giận) có năng lượng cao hơn 'sad' (buồn)).
        # - librosa.feature.rms: Tính Root-Mean-Square (tương đương Energy).
        rms = np.mean(librosa.feature.rms(y=y_filtered).T, axis=0)
        
        # Đặc trưng 3: Zero-Crossing Rate (Tỷ lệ qua điểm 0) (1 giá trị)
        # - Tại sao dùng? ZCR giúp phân biệt âm 'vang' (voiced, ZCR thấp)
        #   với âm 'rít' (unvoiced, như 's', ZCR cao).
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_filtered).T, axis=0)
        
        # 4. Gộp (Stack) các đặc trưng:
        # - Tại sao dùng 'hstack'? Để gộp 3 mảng (mfccs, rms, zcr) thành 1 vector duy nhất.
        #   (20,) + (1,) + (1,) = (22,)
        #   Vector này (dài 22) là đầu vào cho mô hình AI.
        features = np.hstack((mfccs, rms, zcr))
        
        # 5. Lấy nhãn (Label):
        # Đọc nhãn từ tên file (ví dụ: '03') và dịch sang ('happy')
        filename = os.path.basename(file_path)
        label = emotion_map[filename.split('-')[2]]
        
        return features, label
    
    except Exception as e:
        # Bắt lỗi nếu file âm thanh bị hỏng hoặc quá ngắn
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None, None

def run_feature_extraction():
    """
    (Hàm chính cho Bước 1 & 2: Preprocessing & Feature Extraction)
    Hàm này chạy 'extract_features_filtered' cho TẤT CẢ 1440 file âm thanh.
    - Tại sao dùng? Để tạo ra bộ dữ liệu (X, y) và LƯU LẠI.
      Việc trích xuất đặc trưng rất tốn thời gian (vài phút). Chúng ta chỉ chạy 1 lần
      và lưu kết quả vào file 'features_filtered.npy' và 'labels_filtered.npy'.
      Sau đó, mô hình AI (Bước 2) chỉ cần tải file .npy này (mất 1 giây).
    """
    print("--- [BƯỚC 1] BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG (V2 - ĐÃ LỌC) ---")
    all_features = []  # List để chứa các vector (22,)
    all_labels = []    # List để chứa các nhãn ('happy', 'sad'...)
    
    # Tìm tất cả file .wav trong thư mục data/raw/
    file_paths = glob.glob(os.path.join(RAW_DATA_DIR, "Actor_*", "*.wav"))
    
    # 'tqdm' bọc 'file_paths' để tạo thanh tiến trình
    for file_path in tqdm(file_paths, desc="Đang xử lý file âm thanh"):
        features, label = extract_features_filtered(file_path)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
    
    # Chuyển đổi list thành mảng NumPy
    X = np.array(all_features)  # Sẽ có shape (1440, 22)
    y = np.array(all_labels)    # Sẽ có shape (1440,)
    
    print(f"\nTrích xuất hoàn tất. Shape X: {X.shape}, Shape y: {y.shape}")
    
    # Lưu kết quả
    np.save(os.path.join(PROCESSED_DATA_DIR, 'features_filtered.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'labels_filtered.npy'), y)
    
    print("Đã lưu features_filtered.npy và labels_filtered.npy.")
    print("--- [BƯỚC 1] HOÀN THÀNH ---")
    return X, y

# === BƯỚC 2: HÀM AI/ML (Model Training & Evaluation) ===

def run_model_training_and_evaluation(X, y):
    """
    (Hàm chính cho Bước 4 & 5: AI/ML Model & Classification Output)
    Hàm này huấn luyện, đánh giá và lưu mô hình RF V2.
    """
    print("\n--- [BƯỚC 2] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH (RF V2) ---")
    
    # 1. Chuẩn bị dữ liệu cho AI
    
    # a) Chuẩn hóa Đặc trưng (X)
    # - Tại sao dùng 'StandardScaler'? Các đặc trưng có thang đo (scale) rất khác nhau
    #   (MFCCs từ -20 đến +20, ZCR từ 0 đến 1). 'StandardScaler' đưa tất cả về
    #   cùng một thang đo (mean=0, std=1), giúp mô hình AI hoạt động hiệu quả hơn.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # 'fit_transform' vừa học vừa biến đổi
    
    # b) Mã hóa Nhãn (y)
    # - Tại sao dùng 'LabelEncoder'? Mô hình AI không hiểu chữ ('happy', 'sad').
    #   Nó biến đổi 'angry' -> 0, 'calm' -> 1, 'disgust' -> 2,...
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y) # Biến đổi nhãn chữ -> số
    class_names = list(encoder.classes_)
    
    # 2. Chia dữ liệu
    # - Tại sao chia (split)? Chúng ta phải kiểm tra (evaluate) mô hình trên dữ liệu
    #   mà nó 'chưa từng thấy' (tập test).
    # - test_size=0.2: Dùng 20% (288 file) để test, 80% (1152 file) để train.
    # - stratify=y_encoded: Rất quan trọng! Đảm bảo tỷ lệ 8 cảm xúc trong
    #   tập train và test là GIỐNG NHAU (tránh trường hợp xui xẻo là
    #   tất cả file 'neutral' đều rơi vào tập test).
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 3. Huấn luyện (Train) mô hình
    # - Tại sao dùng 'RandomForest'? Đây là mô hình 'ensemble' mạnh mẽ,
    #   ít bị overfitting hơn SVM, và chạy nhanh trên CPU.
    # - n_estimators=100: Sử dụng 100 'cây quyết định' trong 'khu rừng'.
    print("Đang huấn luyện Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) # Đây là lúc mô hình 'học'
    print("Huấn luyện hoàn tất!")

    # 4. Đánh giá (Evaluate) mô hình
    # Dùng mô hình đã học để dự đoán trên tập test (dữ liệu nó chưa thấy)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) # So sánh dự đoán với sự thật
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ (RF V2) ---")
    print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    print("\nBáo cáo Phân loại Chi tiết (Precision, Recall, F1-score):")
    # In ra các chỉ số theo yêu cầu đề bài
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 5. Lưu mô hình (Rất quan trọng cho Demo)
    # - Tại sao lưu? Để 'app.py' (demo) có thể TẢI mô hình đã huấn luyện
    #   mà không cần chạy lại toàn bộ quy trình này.
    print("Đang lưu mô hình (v2), scaler (v2), và encoder (v2)...")
    # Lưu mô hình AI
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_emotion_model_v2.pkl"))
    # Lưu scaler (để app có thể chuẩn hóa file mới)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.pkl"))
    # Lưu encoder (để app có thể dịch số (0) -> chữ ('angry'))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder_v2.pkl"))
    
    print(f"Đã lưu các file vào thư mục {MODEL_DIR}")
    print("--- [BƯỚC 2] HOÀN THÀNH ---")

# --- [PHẦN D] HÀM CHÍNH ĐỂ CHẠY QUY TRÌNH ---

# 'if __name__ == "__main__":' là điểm bắt đầu khi bạn chạy file .py
# (ví dụ: chạy bằng 'python src/main_training_pipeline.py')
if __name__ == "__main__":
    print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN TỔNG THỂ (V2)...")
    
    # CHẠY BƯỚC 1:
    # (Signal Input -> Preprocessing -> Feature Extraction)
    # Kết quả: 2 file .npy được lưu và X_data, y_data được trả về
    X_data, y_data = run_feature_extraction()
    
    # CHẠY BƯỚC 2:
    # (AI/ML Model -> Classification Output)
    # Lấy X_data, y_data từ Bước 1 làm đầu vào
    # Kết quả: 3 file .pkl được lưu và in ra báo cáo
    run_model_training_and_evaluation(X_data, y_data)
    
    print("\nQUY TRÌNH HOÀN TẤT!")