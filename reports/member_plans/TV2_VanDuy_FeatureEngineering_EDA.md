# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 2: VĂN DUY
**Phân công:** `Feature Engineering & EDA`  
**Thời gian thực hiện:** 4 Ngày cốt lõi (Tuần 1 & Đầu Tuần 2)  
**Mục tiêu chính:** Phân tích khám phá dữ liệu (EDA), vector hóa văn bản (TF-IDF & Lexicon features), xử lý mất cân bằng dữ liệu và chia tập Train/Test chuẩn.

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (DAY-BY-DAY CHECKLIST)

### 🟢 Ngày 1: Khám phá Dữ liệu Toàn diện (EDA)
- [ ] Mở và chạy notebook [notebooks/01_data_exploration_eda.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/01_data_exploration_eda.ipynb).
- [ ] **Thực hiện các phân tích thống kê:**
  - Thống kê phân bố số sao đánh giá (1 sao - 5 sao).
  - Phân tích phân bố các nhãn cảm xúc: Tỷ lệ % của `Positive`, `Neutral`, `Negative`.
  - Phân tích độ dài câu (số lượng từ trong `What I liked`, `Suggestions for improvement`).
  - Kiểm tra mức độ tương quan giữa điểm các khía cạnh (Lương thưởng, Đào tạo, Quản lý, Môi trường, OT) với nhãn cảm xúc chung.
- [ ] Xuất và lưu các biểu đồ EDA chất lượng cao vào `reports/figures/` (ví dụ: `eda_rating_distribution.png`, `eda_sentiment_counts.png`).

### 🟢 Ngày 2: Xây dựng Module Trích xuất Đặc trưng (Feature Engineering)
- [ ] Hoàn thiện module [src/features.py](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/src/features.py).
- [ ] **Cấu hình trích xuất đặc trưng văn bản (Text Features):**
  - Sử dụng `TfidfVectorizer` trên trường `clean_advance_text`:
    - Thử nghiệm `ngram_range=(1, 1)` và `ngram_range=(1, 2)`.
    - Thiết lập `max_features` tối ưu (3000 - 5000 từ).
    - Sử dụng `sublinear_tf=True` và lọc bỏ các từ xuất hiện quá ít (`min_df=2`).
  - Lưu lại vectorizer vào file: `models/tfidf_vectorizer.joblib`.

### 🟢 Ngày 3: Ghép đặc trưng số & Xử lý Mất cân bằng dữ liệu (Imbalanced Data)
- [ ] **Chuẩn hóa và ghép đặc trưng số (Numerical Features):**
  - Lấy các thuộc tính số do TV1 tạo ra: `pos_w`, `neg_w`, `sentiment_ratio` và điểm rating thành phần (nếu có).
  - Chuẩn hóa bằng `MinMaxScaler(feature_range=(0, 1))`.
  - Ghép ma trận đặc trưng text và đặc trưng số bằng `scipy.sparse.hstack([X_tfidf, X_num_scaled])`.
- [ ] **Chiến lược chia tập & Xử lý mất cân bằng:**
  - Chia tập dữ liệu Train/Test theo tỷ lệ **80% Train - 20% Test** với kỹ thuật phân tầng `stratify=y` (giữ nguyên tỷ lệ các lớp).
  - Khảo sát và thử nghiệm kỹ thuật cân bằng lớp:
    - Cách 1: Áp dụng `SMOTE` từ thư viện `imbalanced-learn` trên tập Train.
    - Cách 2: Thiết lập `class_weight='balanced'` cho các mô hình.
- [ ] **Bàn giao:** Chuyển giao ma trận đặc trưng $X_{train}, X_{test}, y_{train}, y_{test}$ và file dữ liệu cho **TV3 (Duy Khang)**.

### 🟢 Ngày 4 & Tuần 3: Viết Báo cáo & Kiểm thử chéo
- [ ] Soạn thảo **Mục 2.1, 2.2 (Tổng quan dữ liệu & EDA)** và **Mục 3.1 (Phương pháp trích xuất đặc trưng)** trong báo cáo.
- [ ] Chèn các biểu đồ phân tích EDA vào file báo cáo.
- [ ] Hỗ trợ TV1 rà soát, dọn dẹp code các Jupyter Notebook để đảm bảo chạy mượt từ đầu đến cuối không lỗi runtime.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - File dữ liệu sạch từ TV1: `data/processed/reviews_cleaned.xlsx`.
* **Đầu ra (Outputs bàn giao):**
  - Notebook hoàn chỉnh: [notebooks/01_data_exploration_eda.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/01_data_exploration_eda.ipynb).
  - Module code: [src/features.py](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/src/features.py).
  - File ma trận đặc trưng và bộ vectorizer đã fit: `models/tfidf_vectorizer.joblib`.
  - Toàn bộ hình ảnh biểu đồ EDA trong `reports/figures/`.
  - Nội dung Chương 2.1 - 2.2 và Chương 3.1 của Báo cáo.
