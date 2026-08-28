# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 2: VĂN DUY
**Phân công:** `Feature Engineering & EDA`  
**Thời gian thực hiện:** 4 Ngày cốt lõi (Tuần 1 & Đầu Tuần 2)  
**Mục tiêu chính:** Phân tích khám phá dữ liệu (EDA), xây dựng pipeline NLP text-only bằng TF-IDF, khảo sát lexicon/aspect bằng ablation, xử lý mất cân bằng và bàn giao development/final-test có contract tái lập được.

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (DAY-BY-DAY CHECKLIST)

### 🟢 Ngày 1: Khám phá Dữ liệu Toàn diện (EDA)
- [x] Mở và chạy notebook [notebooks/01_data_exploration_eda.ipynb](../../notebooks/01_data_exploration_eda.ipynb).
- [x] **Thực hiện các phân tích thống kê:**
  - Thống kê phân bố số sao đánh giá (1 sao - 5 sao).
  - Phân tích phân bố các nhãn cảm xúc: Tỷ lệ % của `Positive`, `Neutral`, `Negative`.
  - Phân tích độ dài câu (số lượng từ trong `What I liked`, `Suggestions for improvement`).
  - Kiểm tra mức độ tương quan giữa năm điểm khía cạnh thực có trong dữ liệu với weak label; không suy diễn quan hệ nhân quả và không báo cáo khía cạnh OT vì schema không có trường này.
  - Phân tích phân bố theo công ty/thời gian, text trùng, lexicon coverage và bất nhất giữa `Recommend?` với weak label.
- [x] Xuất và lưu các biểu đồ EDA chất lượng cao vào `reports/figures/` (ví dụ: `eda_rating_distribution.png`, `eda_sentiment_counts.png`).

### 🟢 Ngày 2: Xây dựng Module Trích xuất Đặc trưng (Feature Engineering)
- [x] Hoàn thiện module [src/features.py](../../src/features.py).
- [x] **Cấu hình trích xuất đặc trưng văn bản (Text Features):**
  - Sử dụng `TfidfVectorizer` trên trường `clean_advance_text`:
    - Thử nghiệm `ngram_range=(1, 1)` và `ngram_range=(1, 2)`.
    - Thiết lập `max_features` tối ưu (3000 - 5000 từ).
    - Sử dụng `sublinear_tf=True` và lọc bỏ các từ xuất hiện quá ít (`min_df=2`).
  - Chọn cấu hình bằng 5-fold CV trên development và lưu vectorizer text-only vào `models/text_tfidf_vectorizer.joblib`.

### 🟢 Ngày 3: Ghép đặc trưng số & Xử lý Mất cân bằng dữ liệu (Imbalanced Data)
- [x] **Ablation đặc trưng số (Numerical Features):**
  - Lấy các thuộc tính số do TV1 tạo ra: `pos_w`, `neg_w`, `sentiment_ratio` và điểm rating thành phần (nếu có).
  - Fit `MinMaxScaler` bên trong từng fold development.
  - So sánh text-only, text + lexicon, aspect-only và structured hybrid. Chỉ text-only được bàn giao làm pipeline NLP chính.
- [x] **Chiến lược chia tập & Xử lý mất cân bằng:**
  - Loại text trùng/bất đồng rồi chia **80% development - 20% final test** với `stratify=y`; final test không được dùng để chọn feature.
  - Khảo sát và thử nghiệm kỹ thuật cân bằng lớp:
    - Cách 1: Áp dụng `SMOTE` từ thư viện `imbalanced-learn` trên tập Train.
    - Cách 2: Thiết lập `class_weight='balanced'` cho các mô hình.
- [x] **Bàn giao:** Chuyển giao ma trận đặc trưng $X_{train}, X_{test}, y_{train}, y_{test}$ và file dữ liệu cho **TV3 (Duy Khang)**.

### 🟢 Ngày 4 & Tuần 3: Viết Báo cáo & Kiểm thử chéo
- [x] Soạn thảo **Mục 2.1, 2.2 (Tổng quan dữ liệu & EDA)** và **Mục 3.1 (Phương pháp trích xuất đặc trưng)** trong `reports/eda_feature_engineering.md`.
- [x] Chèn các biểu đồ phân tích EDA vào file báo cáo.
- [ ] Hỗ trợ TV1 rà soát, dọn dẹp code các Jupyter Notebook để đảm bảo chạy mượt từ đầu đến cuối không lỗi runtime.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - File dữ liệu sạch từ TV1: `data/processed/reviews_cleaned.xlsx`.
* **Đầu ra (Outputs bàn giao):**
  - Notebook hoàn chỉnh: [notebooks/01_data_exploration_eda.ipynb](../../notebooks/01_data_exploration_eda.ipynb).
  - Module code: [src/features.py](../../src/features.py).
  - File ma trận đặc trưng và bộ vectorizer đã fit: `models/train_test_features.joblib`, `models/text_feature_extractor.joblib`, `models/text_tfidf_vectorizer.joblib`.
  - Artifact manifest: `models/artifact_manifest.json` và môi trường khóa tại `requirements.lock`.
  - Tài liệu giải thích cho cả nhóm: `reports/overview_for_team.md`.
  - Toàn bộ hình ảnh biểu đồ EDA trong `reports/figures/`.
  - Nội dung Chương 2.1 - 2.2 và Chương 3.1 của Báo cáo.
