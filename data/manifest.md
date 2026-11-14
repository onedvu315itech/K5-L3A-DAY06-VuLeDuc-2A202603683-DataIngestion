# Data Manifest

## 1. Thông tin nguồn dữ liệu

| Trường | Giá trị |
|---|---|
| Document ID | `principles-economics-3e-monopoly` |
| Tên tài liệu | *Principles of Economics 3e – Monopoly* |
| File | `principles-economics-3e-monopoly.pdf` |
| Định dạng | PDF |
| Chủ đề | Economics – Monopoly |
| Nguồn xuất bản | OpenStax |
| Tổng số trang PDF | 50 |
| Số trang nội dung được xử lý | 50 |
| Ngôn ngữ | English |

## 2. Phạm vi sử dụng

- Sử dụng tài liệu làm corpus thử nghiệm cho hệ thống Retrieval-Augmented Generation (RAG).
- Nội dung được trích xuất thành văn bản theo từng trang.
- Dữ liệu được chia chunks theo ba chiến lược:
  - Strategy A: Fixed-size chunking.
  - Strategy B: Paragraph-based chunking.
  - Strategy C: Paragraph-based chunking kết hợp metadata filter.
- Các chunks được embedding và lưu trữ trong PostgreSQL với pgvector.
- Benchmark gồm 30 câu hỏi: 23 Dev và 7 Test.

## 3. Quy trình thu thập và xử lý

| Công đoạn | Mô tả |
|---|---|
| Data source | PDF document |
| Data extraction | Trích xuất văn bản theo từng trang |
| OCR | Chưa xác nhận có thực hiện OCR; parser hiện đánh dấu `needs_ocr` dựa trên trạng thái văn bản trích xuất |
| Chunking | Fixed-size và paragraph-based |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector storage | PostgreSQL + pgvector |

## 4. Quyền sử dụng và giới hạn

- **Nguồn xuất bản:** OpenStax.
- **Giấy phép cụ thể của tài liệu:** Cần xác minh theo trang bản quyền của PDF.
- **Quyền sử dụng trong dự án:** Chỉ sử dụng làm dữ liệu thử nghiệm và đánh giá hệ thống RAG.
- Không mặc định rằng việc tải hoặc sử dụng tài liệu đồng nghĩa với quyền phân phối lại toàn bộ nội dung.
- Khi chia sẻ kết quả, cần ghi nhận nguồn tài liệu và tuân thủ giấy phép áp dụng.

## 5. Data limitations

- Corpus chỉ bao gồm một tài liệu và một chủ đề chính.
- Kết quả benchmark chưa đại diện cho khả năng retrieval trên nhiều lĩnh vực.
- Chất lượng trích xuất phụ thuộc vào cấu trúc và định dạng PDF.
- Thông tin URL tải xuống và giấy phép cụ thể chưa được xác minh.
