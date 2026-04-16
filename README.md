# Dự Án Nhận Diện Rác Thải Bằng YOLOv8

Dự án này xây dựng mô hình nhận diện rác thải trong ảnh bãi rác hoặc môi trường thực tế bằng YOLOv8. Hệ thống có nhiệm vụ phát hiện vị trí vật thể và phân loại chúng thành 5 nhóm chính:

- `plastic`
- `metal`
- `paper`
- `cardboard`
- `trash`

Ngoài phần huấn luyện mô hình, dự án hiện đã có một ứng dụng web demo hoàn chỉnh bằng `Flask + HTML + CSS + JavaScript` để phục vụ báo cáo và trình bày sản phẩm.

## Mục Tiêu Dự Án

- Phát hiện rác thải trong ảnh thực tế
- Phân loại rác theo 5 lớp đã định nghĩa
- Xây dựng quy trình hoàn chỉnh từ xử lý dữ liệu, huấn luyện, fine-tune đến demo suy luận
- Tạo ứng dụng web để trực quan hóa kết quả nhận diện

## Tổng Quan Dữ Liệu

Ban đầu dự án sử dụng bộ dữ liệu YOLO 5 lớp:

- [taco_merged_5class_yolo](./taco_merged_5class_yolo)

Sau đó bổ sung thêm bộ dữ liệu bên ngoài:

- [GARBAGE CLASSIFICATION](./GARBAGE%20CLASSIFICATION)

Bộ dữ liệu này có 6 lớp:

- `BIODEGRADABLE`
- `CARDBOARD`
- `GLASS`
- `METAL`
- `PAPER`
- `PLASTIC`

Để thống nhất với bài toán hiện tại, chỉ giữ lại 4 lớp có thể ánh xạ trực tiếp:

- `PLASTIC -> plastic`
- `METAL -> metal`
- `PAPER -> paper`
- `CARDBOARD -> cardboard`

Hai lớp bị loại bỏ:

- `BIODEGRADABLE`
- `GLASS`

Sau khi remap, dữ liệu được gộp với bộ dữ liệu gốc và tiếp tục được `resplit` lại theo `base image` để tránh hiện tượng trùng ảnh giữa `train`, `valid`, `test`.

## Các Thư Mục Quan Trọng

- [taco_merged_5class_yolo](./taco_merged_5class_yolo)  
Bộ dữ liệu YOLO 5 lớp ban đầu

- [taco_merged_5class_yolo_resplit](./taco_merged_5class_yolo_resplit)  
Bộ dữ liệu gốc sau khi resplit để loại bỏ leakage

- [GARBAGE CLASSIFICATION](./GARBAGE%20CLASSIFICATION)  
Bộ dữ liệu tải thêm trước khi remap class

- [garbage_4class_yolo](./garbage_4class_yolo)  
Bộ dữ liệu ngoài sau khi giữ lại 4 lớp tương thích

- [merged_5class_yolo](./merged_5class_yolo)  
Bộ dữ liệu sau khi gộp bộ cũ và bộ mới

- [merged_5class_yolo_resplit](./merged_5class_yolo_resplit)  
Bộ dữ liệu cuối cùng dùng để fine-tune tiếp

- [runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt](./runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt)  
Model baseline tốt nhất hiện tại, dùng để demo ổn định

- [demo_app](./demo_app)  
Ứng dụng web demo hoàn chỉnh bằng Flask

- [scripts](./scripts)  
Thư mục chứa các script xử lý dữ liệu: phân tích, remap, merge và resplit dataset

## Quy Trình Huấn Luyện

### 1. Phân tích dữ liệu

```bash
python scripts/analyze_yolo_dataset.py
```

### 2. Resplit dữ liệu để tránh leakage

```bash
python scripts/resplit_yolo_dataset.py --source merged_5class_yolo --output merged_5class_yolo_resplit
```

### 3. Huấn luyện baseline YOLOv8n

```bash
yolo detect train data=taco_merged_5class_yolo_resplit/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=8 workers=0 device=cpu project=runs_yolo_baseline name=trash_yolov8n_cpu
```

### 4. Fine-tune trên bộ dữ liệu đã gộp

```bash
yolo detect train data=merged_5class_yolo_resplit/data.yaml model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt imgsz=640 epochs=30 batch=8 workers=0 device=cpu project=runs_yolo_baseline name=trash_yolov8n_finetune_merged
```

### 5. Tiếp tục train nếu bị gián đoạn

```bash
yolo detect train resume model=runs/detect/runs_yolo_baseline/trash_yolov8n_finetune_merged/weights/last.pt
```

## Ứng Dụng Web Demo

Ứng dụng web hiện tại được đặt trong thư mục:

- [demo_app](./demo_app)

### Tính năng chính

- Tải ảnh từ máy tính
- Chụp ảnh trực tiếp bằng webcam
- Chọn model để dự đoán
- Điều chỉnh ngưỡng confidence
- Hiển thị ảnh gốc và ảnh đã nhận diện
- Thống kê số lượng theo từng class
- Hiển thị bảng chi tiết từng bounding box
- Tải ảnh kết quả về máy
- Hỗ trợ chế độ sáng và tối

### Cấu trúc thư mục app

- [demo_app/app.py](./demo_app/app.py)  
Backend Flask xử lý API dự đoán

- [demo_app/models](./demo_app/models)  
Thư mục chứa các model dùng riêng cho phần demo web

- [demo_app/templates/index.html](./demo_app/templates/index.html)  
Giao diện chính

- [demo_app/static/css/styles.css](./demo_app/static/css/styles.css)  
Toàn bộ phần giao diện và theme sáng/tối

- [demo_app/static/js/app.js](./demo_app/static/js/app.js)  
Xử lý upload ảnh, camera, gửi request dự đoán và cập nhật giao diện

- [demo_app/requirements.txt](./demo_app/requirements.txt)  
Các thư viện cần thiết để chạy app

### Cách chạy app

```bash
cd demo_app
python app.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu giao diện chưa cập nhật sau khi chỉnh sửa file, hãy dùng:

```text
Ctrl + F5
```

để tải lại toàn bộ CSS và JavaScript.

## Cách Chạy Demo Bằng Dòng Lệnh

### Dự đoán trên một ảnh

```bash
yolo detect predict model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt source=test.jpg
```

### Dự đoán với ngưỡng confidence

```bash
yolo detect predict model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt source=test.jpg conf=0.25 save_txt=True save_conf=True
```

### Dự đoán bằng webcam

```bash
yolo detect predict model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt source=0
```

### Dự đoán trên video

```bash
yolo detect predict model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt source=video.mp4
```

## Môi Trường Sử Dụng

Môi trường hiện tại của dự án:

- Python `3.12.5`
- Ultralytics `8.4.37`
- PyTorch `2.11.0+cpu`
- Flask `3.1.0`

Cài thư viện cho app web:

```bash
cd demo_app
pip install -r requirements.txt
```

## Các Script Đã Xây Dựng

Các script này được đặt trong thư mục [scripts](./scripts) và là phần rất quan trọng của pipeline xử lý dữ liệu. Chúng giúp dự án không chỉ dừng ở việc train model, mà còn thể hiện rõ cách nhóm làm sạch dữ liệu, tránh leakage và chuẩn hóa đầu vào trước khi fine-tune.

### 1. [analyze_yolo_dataset.py](./scripts/analyze_yolo_dataset.py)

Vai trò:
- Phân tích nhanh chất lượng của một YOLO dataset trước khi huấn luyện.

Script này làm gì:
- Đếm số ảnh, số label và số bounding box trong từng split `train`, `val`, `test`.
- Thống kê số box theo từng class để phát hiện mất cân bằng dữ liệu.
- Kiểm tra các dòng label bị lỗi format, class id sai hoặc ảnh bị thiếu label.
- Phân tích mức độ lặp lại của cùng một `base image`.
- So sánh `base image` giữa các split để phát hiện hiện tượng `data leakage`.

Khi nào dùng:
- Dùng đầu tiên, trước khi train baseline.
- Dùng lại sau mỗi lần tạo dataset mới nếu muốn kiểm tra nhanh độ sạch của dữ liệu.

Đầu vào mặc định:
- `taco_merged_5class_yolo`

Đầu ra:
- Một báo cáo JSON in ra console, phù hợp để lưu log hoặc trích số liệu sang báo cáo đồ án.

### 2. [resplit_yolo_dataset.py](./scripts/resplit_yolo_dataset.py)

Vai trò:
- Chia lại dataset YOLO theo `base image` thay vì chia ngẫu nhiên từng file ảnh.

Vì sao cần script này:
- Dataset export từ Roboflow hoặc qua augment thường có nhiều biến thể của cùng một ảnh gốc.
- Nếu các biến thể này rơi vào nhiều split khác nhau, model có thể đạt metric cao giả tạo.

Script này làm gì:
- Gom các file ảnh thuộc cùng một `base image` vào chung một nhóm.
- Chia các nhóm này vào `train`, `valid`, `test` theo tỉ lệ đặt trước.
- Copy ảnh và label sang một thư mục dataset mới.
- Tạo lại `data.yaml` để dataset đầu ra dùng được ngay với YOLOv8.

Khi nào dùng:
- Dùng sau khi đã kiểm tra thấy dataset có leakage.
- Dùng lại sau khi merge nhiều nguồn dữ liệu vào chung một bộ.

Ví dụ:
```bash
python scripts/resplit_yolo_dataset.py --source merged_5class_yolo --output merged_5class_yolo_resplit
```

### 3. [remap_garbage_dataset_to_4class.py](./scripts/remap_garbage_dataset_to_4class.py)

Vai trò:
- Chuẩn hóa dataset ngoài về hệ class mà dự án đang sử dụng.

Bối cảnh:
- Dataset `GARBAGE CLASSIFICATION` có 6 lớp, nhưng bài toán hiện tại chỉ cần các lớp tương thích trực tiếp với bộ dữ liệu gốc.

Script này làm gì:
- Giữ lại 4 lớp: `PLASTIC`, `METAL`, `PAPER`, `CARDBOARD`.
- Loại bỏ `GLASS` và `BIODEGRADABLE`.
- Đổi `class id` của bộ dữ liệu ngoài sang hệ class mới.
- Bỏ những ảnh không còn bounding box hợp lệ sau bước remap.
- Tạo ra một dataset YOLO mới tên là `garbage_4class_yolo`.

Khi nào dùng:
- Dùng ngay sau khi tải bộ dữ liệu ngoài về và trước bước merge.

Ý nghĩa trong dự án:
- Đây là bước giúp dữ liệu từ nhiều nguồn trở nên thống nhất về ngữ nghĩa class, tránh làm mô hình học bị nhiễu.

### 4. [merge_yolo_datasets.py](./scripts/merge_yolo_datasets.py)

Vai trò:
- Gộp bộ dữ liệu gốc 5 class với bộ dữ liệu ngoài đã remap thành 4 class.

Script này làm gì:
- Đọc dữ liệu từ hai nguồn: dataset gốc và dataset ngoài đã remap.
- Copy ảnh và label sang một thư mục chung.
- Thêm prefix vào tên file để tránh trùng tên giữa hai nguồn dữ liệu.
- Tạo `data.yaml` cho bộ dữ liệu YOLO 5 class mới sau khi gộp.

Khi nào dùng:
- Dùng sau khi đã remap xong dataset ngoài.
- Dùng trước bước `resplit` cuối cùng và trước khi fine-tune model.

Lưu ý:
- Sau khi merge xong vẫn nên chạy `resplit_yolo_dataset.py` thêm một lần nữa để tránh leakage trên bộ dữ liệu mới.

### Thứ tự sử dụng 4 script trong pipeline

1. `analyze_yolo_dataset.py`  
Kiểm tra dataset gốc, phát hiện leakage và vấn đề phân bố class.

2. `remap_garbage_dataset_to_4class.py`  
Chuẩn hóa dataset ngoài về 4 class tương thích.

3. `merge_yolo_datasets.py`  
Gộp dataset ngoài đã remap với dataset gốc.

4. `resplit_yolo_dataset.py`  
Chia lại dataset đã gộp theo `base image` để tạo bộ dữ liệu sạch cho fine-tune.

## Ghi Chú

- Model đang dùng cho demo web đã được gom riêng vào thư mục `demo_app/models` để repo gọn hơn.
- Model mặc định cho demo chính là bản `finetune_best.pt` vì đây là phiên bản cho kết quả tốt hơn so với baseline.
- Phiên fine-tune trên bộ dữ liệu gộp có thể cải thiện thêm, nhưng thời gian huấn luyện lâu hơn do tập dữ liệu lớn hơn đáng kể.
- Muốn cải thiện chất lượng thực tế, cần bổ sung thêm ảnh bãi rác tại đúng bối cảnh triển khai và gán nhãn nhất quán hơn.
- Nếu terminal Windows hiển thị sai dấu tiếng Việt, hãy ưu tiên kiểm tra kết quả trực tiếp trên trình duyệt vì app dùng `UTF-8`.

## Hướng Phát Triển

- Huấn luyện model mạnh hơn như `YOLOv8s`
- Cải thiện nhãn ở các lớp dễ nhầm như `paper`, `cardboard`, `trash`
- Mở rộng ứng dụng web với phần giới thiệu dự án, thông tin nhóm và dashboard trực quan hơn
- Thu thập thêm dữ liệu thực tế tại Việt Nam để tăng khả năng tổng quát hóa
