# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 1: HOÀNG HÔN (TRƯỞNG NHÓM)
**Phân công:** `Business & Data Processing`  
**Thời gian thực hiện:** 3 Ngày cốt lõi (Tuần 1) & Điều phối suốt 3 tuần  
**Mục tiêu chính:** Xác định bài toán nghiệp vụ, xây dựng pipeline tiền xử lý văn bản tiếng Việt chuẩn mực, gán nhãn và tạo bộ dữ liệu sạch.

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (DAY-BY-DAY CHECKLIST)

### 🟢 Ngày 1: Business Objective & Quản lý Tổng thể
- [ ] **Xác định mục tiêu bài toán (Business Understanding):**
  - Làm rõ bài toán phân loại cảm xúc 3 lớp: `Positive`, `Negative`, `Neutral` trên dữ liệu đánh giá ITviec.
  - Thiết kế luồng xử lý tổng thể của hệ thống (Pipeline Architecture).
- [ ] **Khởi tạo và chuẩn hóa môi trường dự án:**
  - Thiết lập repo Git, kiểm tra file `requirements.txt` và phân quyền cho các thành viên.
  - Phân phát kế hoạch chi tiết cho TV2, TV3, TV4.

### 🟢 Ngày 2: Xây dựng Pipeline Tiền xử lý Dữ liệu Text (Data Processing)
- [ ] **Hoàn thiện module `src/preprocessing.py`:**
  - Chuẩn hóa bảng mã Unicode sang chuẩn **NFC** (`unicodedata.normalize`).
  - Xử lý biểu tượng cảm xúc: Ánh xạ emoji/emojicon sang từ ngữ mang cảm xúc tích cực/tiêu cực từ [emojicon.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/emojicon.txt), [positive_emoji.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/positive_emoji.txt), [negative_emoji.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/negative_emoji.txt).
  - Chuẩn hóa teencode / từ viết tắt IT (`cty`, `ot`, `dev`, `pm`,...) bằng [teencode.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/teencode.txt).
  - Chuẩn hóa lỗi chính tả và tiếng Anh bằng [wrong-word.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/wrong-word.txt) và [english-vnmese.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/english-vnmese.txt).
  - Tách từ tiếng Việt (Word Segmentation) bằng `underthesea.word_tokenize`.
  - Lọc bỏ stopwords bằng [vietnamese-stopwords.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/vietnamese-stopwords.txt).
- [ ] **Tạo 2 trường văn bản chuẩn hóa trong DataFrame:**
  - `clean_basic_text`: Chỉ làm sạch ký tự lạ, emoji và teencode (giữ nguyên cấu trúc câu).
  - `clean_advance_text`: Đã tách từ và loại bỏ stopwords.

### 🟢 Ngày 3: Trích xuất đặc trưng Lexicon, Gán nhãn & Xuất dữ liệu
- [ ] **Trích xuất đặc trưng Lexicon:**
  - Đếm số từ tích cực (`pos_w`) và tiêu cực (`neg_w`) dựa trên [positive_words.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/positive_words.txt) và [negative_words.txt](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/dictionaries/negative_words.txt).
  - Đếm số icon tích cực (`pos_e`) và tiêu cực (`neg_e`).
  - Tính toán tỷ lệ cảm xúc: `sentiment_ratio = (pos_w + pos_e - neg_w - neg_e) / (pos_w + pos_e + neg_w + neg_e + 1)`.
- [ ] **Gán nhãn & Kiểm thử chất lượng nhãn:**
  - Gán nhãn cảm xúc 3 lớp (`Positive`, `Neutral`, `Negative`).
  - Xuất file kết quả sạch: `data/processed/reviews_cleaned.xlsx`.
- [ ] **Bàn giao:** Chuyển giao file `reviews_cleaned.xlsx` cho **TV2 (Văn Duy)** và **TV3 (Duy Khang)**.

### 🟢 Tuần 2 & 3: Quản lý, Viết Báo cáo & Thiết kế Slide
- [ ] Họp rà soát tiến độ định kỳ mỗi tuần 2 lần.
- [ ] Soạn thảo **Chương 1 (Tổng quan & Đặt vấn đề)** và **Mục 2.3 (Quy trình tiền xử lý)**.
- [ ] Tổng hợp toàn văn Báo cáo (Word / PDF) theo chuẩn mẫu [final_report_outline.md](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/reports/final_report_outline.md).
- [ ] Thiết kế bộ Slide thuyết trình (15-20 slide) và chủ trì buổi thuyết trình thử (Mock Presentation).

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - Dữ liệu thô: [Reviews.xlsx](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/data/raw/Reviews.xlsx) (8,417 mẫu).
  - Bộ từ điển: `data/dictionaries/` (teencode, stopwords, emoji, wrong-words).
* **Đầu ra (Outputs bàn giao):**
  - Code module hoàn chỉnh: [src/preprocessing.py](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/src/preprocessing.py).
  - Notebook hoàn chỉnh: [notebooks/02_text_preprocessing.ipynb](file:///d:/Trí tuệ nhân tạo/HK2/Xử lý ngôn ngữ tự nhiên/Do_An_Sentiment_Analysis/notebooks/02_text_preprocessing.ipynb).
  - File dữ liệu sạch: `data/processed/reviews_cleaned.xlsx` (có đủ cột `clean_basic_text`, `clean_advance_text`, `pos_w`, `neg_w`, `sentiment_ratio`, `sentiment`).
  - File Báo cáo toàn văn hoàn thiện + File Slide PowerPoint thuyết trình.
