# Dự Án Nhận Diện Rác Thải Bằng YOLOv8

Đây là dự án nhận diện rác thải trong ảnh và video bằng YOLOv8, đồng thời tích hợp sẵn một ứng dụng web demo bằng Flask để phục vụ báo cáo, trình bày sản phẩm và chạy suy luận trực tiếp trên dữ liệu thực tế.

Hệ thống tập trung vào hai bài toán chính:

- phát hiện vị trí vật thể rác trong ảnh hoặc video,
- phân loại chúng vào 5 lớp mục tiêu:
  - `plastic`
  - `metal`
  - `paper`
  - `cardboard`
  - `trash`

Ngoài phần mô hình, project hiện tại đã được tổ chức lại theo hướng gọn hơn: toàn bộ backend Flask, giao diện web, script xử lý dữ liệu và model demo đều nằm ngay ở thư mục gốc để thuận tiện chạy thử và trình bày.

## Tổng Quan Dự Án

Project được xây dựng theo một pipeline hoàn chỉnh, không dừng ở bước train mô hình:

- phân tích chất lượng dataset trước huấn luyện,
- phát hiện và giảm `data leakage` bằng cách `resplit` theo `base image`,
- remap class từ bộ dữ liệu ngoài về hệ class mục tiêu,
- merge nhiều nguồn dữ liệu vào cùng một YOLO dataset,
- fine-tune mô hình YOLOv8n,
- triển khai một web demo để trực quan hóa kết quả suy luận trên ảnh, video và ảnh chụp từ webcam.

Điểm mạnh của project không chỉ nằm ở mô hình, mà còn ở việc nhóm đã kiểm soát rõ pipeline dữ liệu để metric đánh giá đáng tin cậy hơn, thay vì train trực tiếp trên dữ liệu chưa được kiểm tra.

## 5 Lớp Rác Mục Tiêu

Mô hình hiện tại phát hiện và phân loại 5 nhóm rác chính:

- `plastic`: chai nhựa, bao bì nhựa, vật thể nhựa phổ biến
- `metal`: lon, hộp kim loại, vật thể có bề mặt kim loại
- `paper`: giấy vụn, giấy in, vật liệu giấy mỏng
- `cardboard`: thùng carton, bìa cứng, vật liệu đóng gói
- `trash`: các loại rác còn lại không thuộc 4 lớp trên

## Cấu Trúc Thư Mục Hiện Tại

Project hiện tại đã được rút gọn về một cấu trúc thống nhất ở thư mục gốc:

```text
.
├── app.py
├── analyze_yolo_dataset.py
├── merge_yolo_datasets.py
├── remap_garbage_dataset_to_4class.py
├── resplit_yolo_dataset.py
├── requirements.txt
├── test.jpg
├── yolov8n.pt
├── models/
│   ├── baseline_best.pt
│   ├── finetune_best.pt
│   └── finetune_best_git_backup.pt
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── styles.css
    ├── js/
    │   └── app.js
    └── generated/
```

### Ý nghĩa các thành phần chính

- `app.py`  
  Backend Flask xử lý giao diện, API dự đoán ảnh/video và logic suy luận YOLOv8.

- `templates/index.html`  
  Landing page kết hợp trực tiếp với khu vực demo, dùng để trình bày tổng quan dự án và chạy thử mô hình trên cùng một giao diện.

- `static/css/styles.css`  
  Toàn bộ phần giao diện, bố cục, hiệu ứng và theme sáng/tối.

- `static/js/app.js`  
  Xử lý upload ảnh, upload video, mở webcam, gửi request dự đoán và cập nhật kết quả ra giao diện.

- `static/generated/`  
  Nơi lưu các file video đầu vào và video đầu ra đã vẽ bounding box trong quá trình demo.

- `models/`  
  Thư mục chứa các checkpoint dùng cho demo:
  - `baseline_best.pt`: mô hình baseline
  - `finetune_best.pt`: mô hình fine-tune, được chọn làm model mặc định trên web

- `analyze_yolo_dataset.py`, `remap_garbage_dataset_to_4class.py`, `merge_yolo_datasets.py`, `resplit_yolo_dataset.py`  
  Bộ script xử lý dữ liệu phục vụ pipeline huấn luyện.

## Tính Năng Web Demo

Ứng dụng web hiện tại hỗ trợ:

- tải ảnh từ máy tính để nhận diện rác,
- tải video từ máy tính để quét toàn bộ clip,
- chụp ảnh trực tiếp từ webcam,
- chọn model giữa baseline và fine-tuned,
- điều chỉnh ngưỡng `confidence`,
- thiết lập số lượng detection tối đa (`max_det`),
- hiển thị ảnh/video đầu vào và ảnh/video kết quả,
- thống kê số lượng rác theo từng lớp,
- hiển thị bảng chi tiết detection,
- tải ảnh hoặc video kết quả về máy,
- chuyển đổi giao diện sáng/tối,
- trình bày dự án bằng landing page ngay trên cùng giao diện demo.

### Hỗ trợ định dạng

Ảnh:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

Video:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`
- `.m4v`

## Cơ Chế Đếm Trong Video

Phần video của project không đếm theo kiểu cộng dồn tất cả bounding box ở mọi frame. Thay vào đó, ứng dụng quét toàn bộ clip và dùng `tracking` để tránh đếm trùng cùng một vật thể.

Cụ thể:

- video được xử lý bằng `model.track(...)` của Ultralytics,
- tracker dùng cấu hình `bytetrack.yaml`,
- mỗi vật thể được gán một `track ID`,
- nếu cùng một vật thể xuất hiện qua nhiều frame nhưng vẫn giữ nguyên `track ID`, hệ thống chỉ tính đó là `1` đối tượng duy nhất.

Nhờ vậy, thống kê cuối cùng phản ánh gần đúng số lượng vật thể thực sự xuất hiện trong toàn bộ video, thay vì số box lặp lại theo thời gian.

Ngoài tổng số theo lớp, app còn trả về thêm:

- số frame đã quét,
- số frame ước lượng của video,
- FPS,
- thời lượng video,
- tổng số box xuất hiện trên toàn bộ frame,
- tỷ lệ frame có tracking,
- số box chưa được tracker cấp ID và bị bỏ qua khỏi thống kê duy nhất.

## Cách Chạy Web Demo

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
python app.py
```

### 3. Mở trình duyệt

```text
http://127.0.0.1:5000
```

Nếu giao diện chưa cập nhật sau khi chỉnh sửa HTML, CSS hoặc JavaScript, hãy tải cứng lại trình duyệt bằng:

```text
Ctrl + F5
```

## Model Demo Hiện Có

Web app tự động quét hai model trong thư mục `models/`:

- `models/baseline_best.pt`
- `models/finetune_best.pt`

Trong đó:

- `finetune_best.pt` là model mặc định,
- `baseline_best.pt` dùng để so sánh nhanh khi trình bày,
- nếu thiếu cả hai file `.pt`, API sẽ báo lỗi và demo không thể chạy.

## Cách Dùng Demo

### Dự đoán trên ảnh

1. Chọn model
2. Kéo thanh `confidence`
3. Tải ảnh lên hoặc chụp từ webcam
4. Bấm `Quét ngay`
5. Xem:
   - ảnh gốc,
   - ảnh đã nhận diện,
   - số lượng theo lớp,
   - bảng chi tiết bounding box

### Dự đoán trên video

1. Tải video lên
2. Chọn model và ngưỡng confidence
3. Bấm `Quét ngay`
4. Hệ thống sẽ:
   - quét toàn bộ clip,
   - vẽ bounding box lên video đầu ra,
   - thống kê số đối tượng theo `track ID`,
   - lưu kết quả vào `static/generated/`

## Suy Luận Bằng Dòng Lệnh

Nếu muốn thử nhanh mô hình ngoài giao diện web, có thể dùng các lệnh YOLO sau:

### Dự đoán trên một ảnh

```bash
yolo detect predict model=models/finetune_best.pt source=test.jpg
```

### Dự đoán với ngưỡng confidence

```bash
yolo detect predict model=models/finetune_best.pt source=test.jpg conf=0.25 save_txt=True save_conf=True
```

### Dự đoán bằng webcam

```bash
yolo detect predict model=models/finetune_best.pt source=0
```

### Dự đoán trên video

```bash
yolo detect predict model=models/finetune_best.pt source=video.mp4
```

## Quy Trình Xử Lý Dữ Liệu

Một phần quan trọng của project là pipeline xử lý dữ liệu. Nhóm không huấn luyện trực tiếp trên dữ liệu thô mà thực hiện kiểm tra, chuẩn hóa và chia lại dữ liệu để hạn chế leakage.

### 1. Phân tích dataset

Script:

```bash
python analyze_yolo_dataset.py
```

Vai trò:

- đếm số ảnh, số label, số bounding box,
- thống kê phân bố class,
- phát hiện dòng label lỗi format,
- tìm ảnh thiếu label hoặc label thiếu ảnh,
- phát hiện mức độ lặp `base image`,
- kiểm tra chồng lấn `base image` giữa các split `train`, `val`, `test`.

Mặc định script phân tích dataset:

```text
taco_merged_5class_yolo [https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format/data]
```

### 2. Remap bộ dữ liệu ngoài về 4 lớp tương thích

Script:

```bash
python remap_garbage_dataset_to_4class.py
```

Script này đọc từ:

```text
GARBAGE CLASSIFICATION [https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification]
```

và tạo ra:

```text
garbage_4class_yolo
```

Ánh xạ class:

- `PLASTIC -> plastic`
- `METAL -> metal`
- `PAPER -> paper`
- `CARDBOARD -> cardboard`

Hai lớp bị loại bỏ:

- `BIODEGRADABLE`
- `GLASS`

### 3. Gộp dataset

Script:

```bash
python merge_yolo_datasets.py
```

Script gộp:

- `taco_merged_5class_yolo`
- `garbage_4class_yolo`

để tạo ra:

```text
merged_5class_yolo
```

Trong quá trình gộp, script thêm prefix vào tên file để tránh trùng giữa hai nguồn dữ liệu.

### 4. Resplit để giảm leakage

Script:

```bash
python resplit_yolo_dataset.py --source merged_5class_yolo --output merged_5class_yolo_resplit
```

Script này:

- gom ảnh theo `base image`,
- chia lại theo tỷ lệ `train`, `valid`, `test`,
- giữ các phiên bản augment của cùng một ảnh trong cùng một split,
- tạo lại `data.yaml` cho bộ dữ liệu đầu ra.

Các tham số hỗ trợ:

- `--source`
- `--output`
- `--train-ratio`
- `--val-ratio`
- `--test-ratio`
- `--seed`

## Thứ Tự Pipeline Đề Xuất

Nếu bạn có đầy đủ dataset đặt đúng tên thư mục mà các script đang kỳ vọng, quy trình nên chạy theo thứ tự:

1. `python analyze_yolo_dataset.py`
2. `python remap_garbage_dataset_to_4class.py`
3. `python merge_yolo_datasets.py`
4. `python resplit_yolo_dataset.py --source merged_5class_yolo --output merged_5class_yolo_resplit`
5. train baseline YOLOv8n
6. fine-tune trên bộ dữ liệu đã gộp và resplit

## Lệnh Huấn Luyện Tham Khảo

### Train baseline YOLOv8n

```bash
yolo detect train data=taco_merged_5class_yolo_resplit/data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=8 workers=0 device=cpu project=runs_yolo_baseline name=trash_yolov8n_cpu
```

### Fine-tune trên bộ dữ liệu đã gộp

```bash
yolo detect train data=merged_5class_yolo_resplit/data.yaml model=runs/detect/runs_yolo_baseline/trash_yolov8n_cpu/weights/best.pt imgsz=640 epochs=30 batch=8 workers=0 device=cpu project=runs_yolo_baseline name=trash_yolov8n_finetune_merged
```

### Tiếp tục train nếu bị gián đoạn

```bash
yolo detect train resume model=runs/detect/runs_yolo_baseline/trash_yolov8n_finetune_merged/weights/last.pt
```

## Môi Trường Và Thư Viện

Các thư viện chính trong project hiện tại:

- `Flask 3.1.0`
- `Ultralytics 8.4.37`
- `Pillow`
- `NumPy`
- `OpenCV`
- `imageio-ffmpeg`
- `lap`

File cài đặt:

```text
requirements.txt
```

Khuyến nghị dùng:

- Python `3.12+`
- môi trường ảo `venv`

## Một Số Lưu Ý

- `static/generated/` sẽ phát sinh thêm file trong quá trình demo video.
- `finetune_best.pt` đang là model mặc định để trình bày vì cho kết quả tốt hơn baseline.
- `yolov8n.pt` được giữ trong project để thuận tiện cho việc huấn luyện baseline.
- Nếu terminal Windows hiển thị sai dấu tiếng Việt, hãy ưu tiên đọc nội dung trực tiếp trên trình duyệt hoặc editor đang dùng `UTF-8`.

## Hướng Phát Triển

- huấn luyện thêm các backbone mạnh hơn như `YOLOv8s`
- bổ sung dữ liệu thực tế tại Việt Nam để tăng khả năng tổng quát hóa
- cải thiện chất lượng nhãn ở các lớp dễ nhầm như `paper`, `cardboard`, `trash`
- mở rộng phần dashboard trên web để trình bày metric và kết quả trực quan hơn
- tối ưu thêm tốc độ xử lý video nếu cần chạy trên tập clip dài hơn
