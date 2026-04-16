from __future__ import annotations

import base64
import io
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image
from ultralytics import YOLO


APP_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = APP_ROOT / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

MODEL_CANDIDATES = [
    (
        "baseline",
        "YOLOv8n Baseline",
        APP_ROOT / "models" / "baseline_best.pt",
    ),
    (
        "finetune_latest",
        "YOLOv8n Fine-tuned",
        APP_ROOT / "models" / "finetune_best.pt",
    ),
]

DEFAULT_CONFIDENCE = 0.25
DEFAULT_MAX_DET = 50


app = Flask(__name__)
app.json.ensure_ascii = False
_MODEL_CACHE: dict[str, YOLO] = {}


def available_models() -> list[dict]:
    models = []
    for model_id, label, path in MODEL_CANDIDATES:
        if path.exists():
            models.append(
                {
                    "id": model_id,
                    "label": label,
                    "path": str(path),
                    "is_default": model_id == "finetune_latest",
                }
            )
    return models


def get_model(model_id: str) -> tuple[YOLO, dict]:
    models = available_models()
    if not models:
        raise FileNotFoundError("Không tìm thấy file model .pt để demo.")

    selected = next((item for item in models if item["id"] == model_id), models[0])
    cache_key = selected["path"]
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = YOLO(cache_key)
    return _MODEL_CACHE[cache_key], selected


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def frame_to_base64(frame: np.ndarray) -> str:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return image_to_base64(Image.fromarray(rgb_frame))


def sort_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def resolve_media_kind(filename: str, mimetype: str = "") -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in VIDEO_EXTS or mimetype.startswith("video/"):
        return "video"
    if suffix in IMAGE_EXTS or mimetype.startswith("image/"):
        return "image"
    raise ValueError("Định dạng file chưa được hỗ trợ. Vui lòng chọn ảnh hoặc video phổ biến.")


def parse_image_result(result) -> tuple[Image.Image, list[dict], dict[str, int]]:
    plotted = result.plot()
    annotated_image = Image.fromarray(plotted[:, :, ::-1])

    detections = []
    counts: dict[str, int] = {}
    names = result.names
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = names[class_id]
            confidence = round(float(box.conf.item()), 4)
            xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            counts[class_name] = counts.get(class_name, 0) + 1
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": xyxy,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return annotated_image, detections, sort_counts(counts)


def build_image_payload(uploaded_file, confidence: float, max_det: int, model_id: str) -> dict:
    image = Image.open(uploaded_file.stream).convert("RGB")
    model, selected_model = get_model(model_id)
    results = model.predict(
        source=np.array(image),
        conf=confidence,
        max_det=max_det,
        device="cpu",
        verbose=False,
    )
    annotated_image, detections, counts = parse_image_result(results[0])

    return {
        "ok": True,
        "media_type": "image",
        "model_label": selected_model["label"],
        "original_image": image_to_base64(image),
        "annotated_image": image_to_base64(annotated_image),
        "detections": detections,
        "counts": counts,
        "summary": {
            "total_detections": len(detections),
            "classes_found": len(counts),
            "note": "Kết quả hiển thị trên một ảnh đầu vào.",
        },
    }


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.filename).suffix.lower() or ".mp4"
    saved_path = GENERATED_DIR / f"{uuid4().hex}_input{suffix}"
    uploaded_file.save(saved_path)
    return saved_path


def ensure_browser_video(raw_video_path: Path) -> Path:
    output_video_path = raw_video_path.with_name(f"{raw_video_path.stem}_browser.mp4")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_video_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_video_path


def build_video_payload(uploaded_file, confidence: float, max_det: int, model_id: str) -> dict:
    input_video_path = save_uploaded_video(uploaded_file)
    raw_output_video_path = GENERATED_DIR / f"{input_video_path.stem}_annotated_raw.mp4"

    model, selected_model = get_model(model_id)
    model.predictor = None

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise ValueError("Không thể mở video đã tải lên.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = round(float(fps), 2) if fps and fps > 0 else 24.0
    estimated_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    success, frame = capture.read()
    if not success:
        capture.release()
        raise ValueError("Video không có frame hợp lệ để xử lý.")

    height, width = frame.shape[:2]
    original_preview = frame_to_base64(frame)
    writer = cv2.VideoWriter(
        str(raw_output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    capture.release()

    unique_track_ids_by_class: defaultdict[str, set[int]] = defaultdict(set)
    frame_level_counts: Counter[str] = Counter()
    processed_frames = 0
    annotated_preview = ""
    tracking_frames = 0
    frames_with_boxes = 0
    untracked_boxes = 0

    # Dùng track() trực tiếp trên cả video để tracker giữ ID ổn định xuyên suốt clip.
    results = model.track(
        source=str(input_video_path),
        conf=confidence,
        max_det=max_det,
        device="cpu",
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for result in results:
        processed_frames += 1
        annotated_frame = result.plot()
        writer.write(annotated_frame)

        if processed_frames == 1:
            annotated_preview = frame_to_base64(annotated_frame)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            frames_with_boxes += 1
            class_ids = boxes.cls.int().tolist()
            if boxes.id is not None:
                track_ids = boxes.id.int().tolist()
                tracking_frames += 1
            else:
                track_ids = [None] * len(class_ids)

            for index, class_id in enumerate(class_ids):
                class_name = result.names[int(class_id)]
                frame_level_counts[class_name] += 1

                track_id = track_ids[index]
                if track_id is None:
                    # Không cộng vào thống kê duy nhất nếu tracker chưa cấp ID,
                    # như vậy sẽ tránh việc cùng một vật thể bị đếm lặp qua nhiều frame.
                    untracked_boxes += 1
                    continue
                unique_track_ids_by_class[class_name].add(int(track_id))

    writer.release()

    if processed_frames == 0:
        raise ValueError("Không đọc được frame nào từ video.")

    try:
        output_video_path = ensure_browser_video(raw_output_video_path)
        raw_output_video_path.unlink(missing_ok=True)
    except Exception:
        output_video_path = raw_output_video_path

    unique_counts = sort_counts(
        {class_name: len(track_ids) for class_name, track_ids in unique_track_ids_by_class.items()}
    )
    duration_seconds = round(processed_frames / fps, 2) if fps else 0
    frame_level_total = sum(frame_level_counts.values())
    tracking_ratio = round((tracking_frames / processed_frames) * 100, 2) if processed_frames else 0

    video_summary = {
        "source_filename": uploaded_file.filename,
        "processed_frames": processed_frames,
        "estimated_frame_count": estimated_frame_count,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "frame_level_detections": frame_level_total,
        "frames_with_boxes": frames_with_boxes,
        "tracking_coverage_percent": tracking_ratio,
        "untracked_boxes_ignored": untracked_boxes,
        "counting_mode_label": "Mỗi track ID chỉ được đếm một lần trên toàn video",
    }

    return {
        "ok": True,
        "media_type": "video",
        "model_label": selected_model["label"],
        "original_preview_image": original_preview,
        "annotated_preview_image": annotated_preview,
        "original_video_url": url_for("static", filename=f"generated/{input_video_path.name}"),
        "annotated_video_url": url_for("static", filename=f"generated/{output_video_path.name}"),
        "counts": unique_counts,
        "detections": [],
        "video_summary": video_summary,
        "summary": {
            "total_detections": sum(unique_counts.values()),
            "classes_found": len(unique_counts),
            "note": "Mỗi vật thể chỉ được tính một lần trên toàn video dựa trên track ID duy nhất.",
        },
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=available_models(),
        default_confidence=DEFAULT_CONFIDENCE,
        default_max_det=DEFAULT_MAX_DET,
    )


@app.post("/api/predict")
def predict():
    try:
        uploaded_file = request.files.get("media") or request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"ok": False, "error": "Vui lòng chọn ảnh hoặc video để dự đoán."}), 400

        confidence = float(request.form.get("conf", DEFAULT_CONFIDENCE))
        max_det = int(request.form.get("max_det", DEFAULT_MAX_DET))
        model_id = request.form.get("model_id", "finetune_latest")
        media_kind = resolve_media_kind(uploaded_file.filename, uploaded_file.mimetype or "")

        if media_kind == "video":
            payload = build_video_payload(uploaded_file, confidence, max_det, model_id)
        else:
            payload = build_image_payload(uploaded_file, confidence, max_det, model_id)
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Lỗi dự đoán: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
