# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 4: THÀNH TRUNG
**Phân công:** `Evaluation, Sentiment Insights & Deployment`  
**Thời gian thực hiện:** 4 Ngày cốt lõi (Tuần 2 & Tuần 3)  
**Mục tiêu chính:** Đánh giá toàn diện các mô hình (Confusion Matrix, Error Analysis), trích xuất Insight cảm xúc doanh nghiệp (WordCloud), xây dựng ứng dụng Web Demo (Streamlit/Gradio) và hoàn thiện các chương kết quả trong báo cáo.

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (DAY-BY-DAY CHECKLIST)

### 🟢 Ngày 1: Đánh giá Mô hình & Phân tích Lỗi sai (Evaluation & Error Analysis)
- [ ] Nhận kết quả dự đoán của các mô hình từ **TV3 (Duy Khang)**.
- [ ] **Tính toán đầy đủ các thang đo đánh giá trên tập Test:**
  - **Accuracy** (Độ chính xác toàn diện).
  - **Precision, Recall, F1-score** theo từng lớp: `Positive`, `Neutral`, `Negative`.
  - **Macro F1-score** (Đánh giá công bằng giữa các lớp) & **Weighted F1-score**.
- [ ] **Vẽ ma trận nhầm lẫn (Confusion Matrix):**
  - Dùng `seaborn.heatmap` vẽ Confusion Matrix cho từng mô hình (Naive Bayes, Logistic Regression, SVM, Stacking, ViSoBERT).
  - Lưu toàn bộ ảnh biểu đồ vào thư mục `reports/figures/` (ví dụ: `cm_stacking_model.png`).
- [ ] **Phân tích lỗi sai chuyên sâu (Error Analysis):**
  - Trích xuất 10-15 câu mẫu mà mô hình đoán sai (ví dụ: Nhãn thật là Negative nhưng mô hình đoán Positive).
  - Phân tích nguyên nhân: Do câu châm biếm ("Công ty tuyệt vời, suốt ngày được OT xuyên đêm không lương!"), câu phủ định ("Không thể không khen"), hoặc do câu ngắn thiếu ngữ cảnh.

### 🟢 Ngày 2: Trích xuất Insight Cảm xúc Doanh nghiệp & WordCloud
- [ ] Mở và chạy notebook [notebooks/05_company_sentiment_insights.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/05_company_sentiment_insights.ipynb).
- [ ] **Tạo WordCloud cảm xúc toàn diện:**
  - Tạo WordCloud cho toàn bộ tập đánh giá Tích cực (`wordcloud_positive_all.png`).
  - Tạo WordCloud cho toàn bộ tập đánh giá Tiêu cực (`wordcloud_negative_all.png`).
- [ ] **Phân tích Case Study theo từng Doanh nghiệp cụ thể:**
  - Lọc dữ liệu theo top 2-3 công ty có nhiều đánh giá nhất trên ITviec (ví dụ: FPT Software, Hitachi Digital Services, VNG, Viettel,...).
  - Thống kê tỷ lệ phần trăm đánh giá Tích cực / Tiêu cực tại công ty đó.
  - Tạo WordCloud riêng về các vấn đề bị phàn nàn nhiều nhất (Điểm yếu cần cải thiện) và các điểm được khen ngợi nhiều nhất (Điểm mạnh) của công ty.
  - Đưa ra đề xuất cải tiến thiết thực cho Ban lãnh đạo & HR của doanh nghiệp.

### 🟢 Ngày 3: Xây dựng Ứng dụng Web Demo Phân loại Cảm xúc (Deployment)
- [ ] **Xây dựng ứng dụng Web tương tác bằng Streamlit hoặc Gradio:**
  - Tạo file `app.py` trong thư mục gốc.
  - Tải mô hình tốt nhất (`models/best_sentiment_model.joblib`) và bộ `models/tfidf_vectorizer.joblib`.
  - Giao diện gồm:
    - **Ô nhập văn bản:** Cho phép người dùng nhập 1 câu review bất kỳ.
    - **Nút "Dự đoán Cảm xúc":** Hệ thống tự động tiền xử lý (qua `TextPreprocessor`) $\rightarrow$ TF-IDF $\rightarrow$ Mô hình dự đoán.
    - **Hiển thị kết quả:** Nhãn cảm xúc (`Tích cực`, `Tiêu cực`, `Trung tính`) kèm xác suất phần trăm (Confidence Score) và icon tương ứng.
    - **Tab Dashboard:** Hiển thị biểu đồ phân tích cảm xúc và WordCloud của các công ty IT.
- [ ] Chạy thử nghiệm cục bộ và chụp ảnh màn hình Demo để đưa vào Slide và Báo cáo.

### 🟢 Ngày 4: Hoàn thiện các Chương Báo cáo (Chương 4, 5, 6)
- [ ] Soạn thảo **Chương 4: Kết quả thực nghiệm & Đánh giá mô hình** (chèn bảng so sánh, Confusion Matrix, Error Analysis).
- [ ] Soạn thảo **Chương 5: Phân tích Insight cảm xúc doanh nghiệp & Triển khai Demo**.
- [ ] Soạn thảo **Chương 6: Kết luận & Hướng phát triển**.
- [ ] Bàn giao toàn bộ nội dung cho **TV1 (Hoàng Hôn)** để tổng hợp báo cáo hoàn chỉnh.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - Mảng dự đoán `y_pred` và file mô hình từ TV3.
  - File dữ liệu sạch từ TV1 & TV2.
* **Đầu ra (Outputs bàn giao):**
  - Notebook hoàn chỉnh: [notebooks/05_company_sentiment_insights.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/05_company_sentiment_insights.ipynb).
  - Ứng dụng Web Demo tương tác (`app.py`).
  - Toàn bộ ảnh Confusion Matrix và WordCloud trong `reports/figures/`.
  - Bảng phân tích lỗi sai (Error Analysis).
  - Nội dung Chương 4, 5, 6 của Báo cáo.
