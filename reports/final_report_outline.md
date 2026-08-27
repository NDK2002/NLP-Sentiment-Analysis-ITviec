# ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN NLP: PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ ITVIEC
*(Chuyên sâu: Sentiment Analysis & Employee Feedback Analytics)*

---

## CHƯƠNG 1: TỔNG QUAN VÀ ĐẶT VẤN ĐỀ
1.1. Bối cảnh đề tài và tầm quan trọng của việc phân tích ý kiến đánh giá nhân sự trong ngành CNTT.  
1.2. Mục tiêu nghiên cứu:
   - Xây dựng hệ thống tự động phân loại cảm xúc đánh giá (Tích cực, Tiêu cực, Trung tính).
   - So sánh hiệu năng giữa các thuật toán Machine Learning truyền thống, mô hình Ensemble và mô hình Pretrained Transformer.
   - Khám phá các chủ đề/từ khóa then chốt tác động đến cảm xúc nhân viên theo từng công ty.  
1.3. Bố cục của báo cáo.

---

## CHƯƠNG 2: TỔNG QUAN DỮ LIỆU & TIỀN XỬ LÝ (DATA & PREPROCESSING)
2.1. Giới thiệu bộ dữ liệu ITviec Reviews (Cấu trúc bảng, các trường nội dung review, điểm đánh giá thành phần).  
2.2. Khám phá dữ liệu (EDA):
   - Phân tích phân bố số sao rating và mức độ mất cân bằng lớp.
   - Thống kê độ dài văn bản đánh giá, phân bố từ ngữ.
   - Phân tích tương quan giữa điểm các khía cạnh (Lương thưởng, Đào tạo, Quản lý, Môi trường, OT) với cảm xúc chung.  
2.3. Quy trình tiền xử lý văn bản tiếng Việt chuyên sâu:
   - Chuẩn hóa mã Unicode NFC.
   - Xử lý biểu tượng cảm xúc (Emoji/Emojicon) thành từ ngữ mang sắc thái.
   - Chuẩn hóa viết tắt (teencode), thuật ngữ IT và sửa lỗi chính tả.
   - Tách từ tiếng Việt (Word Segmentation) bằng `underthesea`.
   - Lọc bỏ stopwords tiếng Việt.
2.4. Chiến lược gán nhãn cảm xúc và tạo tập dữ liệu huấn luyện.

---

## CHƯƠNG 3: BIỂU DIỄN VĂN BẢN VÀ MÔ HÌNH HỌC MÁY (MODELING)
3.1. Phương pháp trích xuất đặc trưng văn bản:
   - TF-IDF Vectorizer (N-gram 1-2, sublinear TF, phân tích tham số tối ưu).
   - Trích xuất đặc trưng Lexicon cảm xúc (đếm từ tích cực/tiêu cực).
   - Biểu diễn ngữ cảnh với Pretrained Language Model (ViSoBERT / PhoBERT).
3.2. Thiết kế các mô hình học máy:
   - Mô hình 1: Multinomial Naive Bayes (Baseline).
   - Mô hình 2: Logistic Regression (với trọng số lớp cân bằng).
   - Mô hình 3: Support Vector Machine (Linear SVM).
   - Mô hình 4: Random Forest Classifier.
   - Mô hình 5: Stacking Ensemble Classifier.
   - Mô hình 6: Fine-tuned ViSoBERT (Deep Learning).
3.3. Kỹ thuật xử lý mất cân bằng dữ liệu và tinh chỉnh siêu tham số (Hyperparameter Tuning với K-Fold Cross Validation).

---

## CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ (EVALUATION)
4.1. Môi trường thực nghiệm và các thang đo đánh giá (Accuracy, Macro F1, Weighted F1, Precision, Recall).  
4.2. Bảng tổng hợp so sánh hiệu năng giữa các mô hình.  
4.3. Phân tích ma trận nhầm lẫn (Confusion Matrix).  
4.4. Phân tích lỗi sai chuyên sâu (Error Analysis):
   - Phân tích các trường hợp mô hình đoán sai (câu châm biếm, phủ định kép, câu chứa cả ý khen và chê).
   - Đánh giá ảnh hưởng của bước tiền xử lý đối với độ chính xác của mô hình.

---

## CHƯƠNG 5: PHÂN TÍCH INSIGHT CẢM XÚC DOANH NGHIỆP & TRIỂN KHAI (BUSINESS INSIGHTS & DEMO)
5.1. Trực quan hóa đám mây từ khóa (WordCloud) cảm xúc Tích cực và Tiêu cực.  
5.2. Phân tích cảm xúc theo từng doanh nghiệp (Case Study 2-3 công ty IT tiêu biểu):
   - Tỷ lệ hài lòng / không hài lòng của nhân viên.
   - Các chủ đề được khen ngợi nhiều nhất (Điểm mạnh của công ty).
   - Các vấn đề bị phàn nàn nhiều nhất (Chế độ OT, Lương thưởng, Quy trình quản lý).  
5.3. Xây dựng ứng dụng Demo trực quan (Gradio / Streamlit) cho bài toán phân loại cảm xúc.  
5.4. Đề xuất giải pháp thực tế cho ban quản lý doanh nghiệp và phòng nhân sự (HR).

---

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
6.1. Những kết quả chính đạt được của đề tài.  
6.2. Những điểm hạn chế còn tồn tại.  
6.3. Hướng phát triển trong tương lai (Phân tích cảm xúc theo từng khía cạnh chi tiết - Aspect-Based Sentiment Analysis ABSA, ứng dụng LLM Agent).
