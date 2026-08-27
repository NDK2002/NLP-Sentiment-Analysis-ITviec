# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 3: DUY KHANG
**Phân công:** `Modeling & Hyperparameter Tuning`  
**Thời gian thực hiện:** 4 Ngày cốt lõi (Tuần 2)  
**Mục tiêu chính:** Xây dựng, huấn luyện và tối ưu ít nhất 4 mô hình Machine Learning, mô hình Stacking Ensemble và thử nghiệm mô hình Pretrained ViSoBERT.

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (DAY-BY-DAY CHECKLIST)

### 🟢 Ngày 1: Huấn luyện 4 Mô hình Machine Learning Cơ sở (Base Models)
- [ ] Nhận tập đặc trưng $X_{train}, X_{test}, y_{train}, y_{test}$ từ **TV2 (Văn Duy)**.
- [ ] Mở và làm việc trên [notebooks/03_sentiment_modeling_ml.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/03_sentiment_modeling_ml.ipynb) và [src/models.py](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/src/models.py).
- [ ] **Khởi tạo và huấn luyện 4 mô hình Supervised Learning:**
  1. **Multinomial Naive Bayes:** Mô hình Baseline cổ điển cho phân loại văn bản (`MultinomialNB(alpha=1.0)`).
  2. **Logistic Regression:** Mô hình hồi quy logistic tuyến tính mạnh mẽ (`LogisticRegression(max_iter=1000, class_weight='balanced')`).
  3. **Support Vector Machine (Linear SVM):** Mô hình SVM tối ưu cho vector TF-IDF không gian nhiều chiều (`SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True)`).
  4. **Random Forest Classifier / Gradient Boosting:** Mô hình cây quyết định kết hợp (`RandomForestClassifier(n_estimators=100, class_weight='balanced')`).

### 🟢 Ngày 2: Tinh chỉnh Siêu tham số (Hyperparameter Tuning với K-Fold CV)
- [ ] Sử dụng `GridSearchCV` hoặc `RandomizedSearchCV` kết hợp **5-Fold Cross Validation** để tìm bộ siêu tham số tối ưu cho từng mô hình:
  - **Naive Bayes:** Tinh chỉnh `alpha` $\in [0.01, 0.1, 0.5, 1.0, 2.0]$.
  - **Logistic Regression:** Tinh chỉnh `C` $\in [0.1, 1.0, 5.0, 10.0]$, `solver=['lbfgs', 'saga']`, `penalty=['l1', 'l2']`.
  - **Linear SVM:** Tinh chỉnh `C` $\in [0.1, 0.5, 1.0, 2.0, 5.0]$.
  - **Random Forest:** Tinh chỉnh `n_estimators=[100, 200]`, `max_depth=[10, 20, None]`.
- [ ] Ghi nhận bảng điểm số Cross-Validation và điểm số trên tập Test của các mô hình sau khi tuning.

### 🟢 Ngày 3: Xây dựng Mô hình Nâng cao (Stacking Ensemble & ViSoBERT)
- [ ] **Mô hình kết hợp Stacking Classifier:**
  - Kết hợp 3 mô hình cơ sở tốt nhất (`Naive Bayes`, `Logistic Regression`, `Linear SVM`) làm Base Estimators.
  - Sử dụng `Logistic Regression` làm Final Estimator (Meta-Classifier) để tổng hợp trọng số dự đoán.
- [ ] **Thử nghiệm Pretrained ViSoBERT (Deep Learning):**
  - Mở [notebooks/04_sentiment_modeling_deeplearning.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/04_sentiment_modeling_deeplearning.ipynb).
  - Tải mô hình `5CD-AI/Vietnamese-Sentiment-visobert` từ HuggingFace Transformers.
  - Chạy dự đoán trên tập test để lấy điểm Benchmark so sánh với Machine Learning truyền thống.
- [ ] **Lưu trữ mô hình:**
  - Lưu file mô hình có kết quả tốt nhất vào: `models/best_sentiment_model.joblib`.
- [ ] **Bàn giao:** Chuyển giao toàn bộ kết quả dự đoán (`y_pred_nb`, `y_pred_lr`, `y_pred_svm`, `y_pred_stack`, `y_pred_visobert`) cho **TV4 (Thành Trung)**.

### 🟢 Ngày 4 & Tuần 3: Viết Báo cáo & Hoàn thiện Số liệu
- [ ] Soạn thảo **Mục 3.2 (Thiết kế mô hình)** và **Mục 3.3 (Kỹ thuật tinh chỉnh siêu tham số)** trong báo cáo.
- [ ] Xuất bảng so sánh hiệu năng tổng hợp (Accuracy, F1-Score) và vẽ biểu đồ cột so sánh F1-Score giữa các mô hình.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - Ma trận đặc trưng $X_{train}, X_{test}, y_{train}, y_{test}$ từ TV2.
* **Đầu ra (Outputs bàn giao):**
  - Notebook hoàn chỉnh: [notebooks/03_sentiment_modeling_ml.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/03_sentiment_modeling_ml.ipynb) & [04_sentiment_modeling_deeplearning.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/04_sentiment_modeling_deeplearning.ipynb).
  - Module code: [src/models.py](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/src/models.py).
  - File mô hình đã huấn luyện: `models/best_sentiment_model.joblib`.
  - Mảng dự đoán `y_pred` của các mô hình trên tập test gửi cho TV4.
  - Nội dung Chương 3.2 - 3.3 của Báo cáo.
