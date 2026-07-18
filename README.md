## 📝 Tổng quan về Luồng Xử lý của Dự án

Quy trình trong Notebook được thực thi chặt chẽ qua **5 khối chức năng chính** sau:

### 1. Kỹ nghệ Đặc trưng Bảng & Chia K-Fold (Tabular Feature Engineering & CV Split)
* **Xử lý giá trị thiếu:** Tự động điền giá trị trung vị (median) cho các biến số và điền giá trị xuất hiện nhiều nhất (mode) cho các biến phân loại[cite: 3].
* **Mô phỏng dấu hiệu "Vịt con xấu xí" (Ugly Duckling Sign):** Tạo ra các đặc trưng chuẩn hóa theo từng bệnh nhân bằng thuật toán toán học để phát hiện các tổn thương dị biệt so với nền da của chính họ[cite: 3].
* **Chia tập dữ liệu an toàn:** Tạo phân tách cấu trúc **5-fold Stratified Group K-Fold** dựa trên `patient_id` nhằm chống rò rỉ dữ liệu (data leakage) giữa tập huấn luyện và tập kiểm thử[cite: 3].

### 2. Khởi tạo Dataset & Bộ Giảm Xóc Lỗi Ảnh (Dataset & Image Handler)
* **Đọc dữ liệu tối ưu:** Định nghĩa lớp PyTorch `ISICDataset` tùy chỉnh, sử dụng thư viện `h5py` để stream trực tiếp dữ liệu ảnh từ file định dạng `.hdf5` mà không làm tràn RAM[cite: 3].
* **Tích hợp khiên chống sập code:** Bổ sung bộ xử lý lỗi `KeyError` nhằm tự động thay thế các ảnh bị thiếu bằng các tensor ma trận 0 (ảnh đen rỗng), đảm bảo pipeline chạy liên tục đến cuối[cite: 3].
* **Tiền xử lý & Tăng cường ảnh:** Thực hiện chuẩn hóa kênh màu và thay đổi kích thước ảnh về chuẩn $224 \times 224$ thông qua thư viện nâng cao `albumentations`[cite: 3].

### 3. Trích xuất Đặc trưng Hình ảnh Đa Kiến trúc (NetX)
* **Học sâu đa nền tảng:** Tận dụng sức mạnh của các mô hình pre-trained hàng đầu từ thư viện `timm` để nhận diện các cấu trúc hình ảnh phức tạp[cite: 3].
* **Tích hợp GeM Pooling:** Thay thế lớp Pooling mặc định của mô hình bằng **Generalized Mean (GeM) Pooling** để chủ động giữ lại các dị thường siêu nhỏ ở biên vết thương thay vì bị làm mờ bởi ảnh nền[cite: 3].
* **Tạo đặc trưng Out-of-Fold (OOF):** Trích xuất xác suất dự đoán OOF bằng 3 kiến trúc mạng song song: `ConvNeXtV2-Nano`, `ViT-Tiny`, và `ViT-Small` chạy xuyên suốt trên cả 5 folds dữ liệu[cite: 3].

### 4. Huấn luyện Mô hình Cây Tổ hợp (Soft Voting Ensemble)
* **Thanh lọc dữ liệu (Anti-Leakage):** Loại bỏ toàn bộ các ID định danh, các cột chứa thông tin chẩn đoán phụ hoặc bất kỳ cột nào có nguy cơ rò rỉ đáp án trước khi đưa vào mô hình GBDT[cite: 3].
* **Huấn luyện đa thuật toán:** Chạy song song 3 thuật toán Gradient Boosted Decision Tree (GBDT) mạnh mẽ nhất hiện nay trên tập dữ liệu đã được dung hợp muộn (Late Fusion): `LightGBM`, `XGBoost`, và `CatBoost`[cite: 3].
* **Tổ hợp Soft Voting:** Trộn các dự đoán OOF của cả 3 mô hình cây bằng phương pháp bỏ phiếu mềm (soft voting) với trọng số đều nhau để giảm thiểu phương sai và tăng độ ổn định[cite: 3].

### 5. Đánh giá Mô hình (Evaluation)
* **Chấm điểm theo chuẩn cuộc thi:** Đánh giá hiệu năng thực tế của hệ thống bằng chỉ số **partial Area Under the ROC Curve (pAUC)** với ngưỡng True Positive Rate (TPR) tối thiểu từ 80% trở lên[cite: 3].
