from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO


APP_ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    (
        "baseline",
        "YOLOv8n Baseline",
        APP_ROOT / "models" / "baseline_best.pt",
    ),
    (
        "finetune",
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
                    "is_default": model_id == "finetune",
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


def parse_result(result) -> tuple[Image.Image, list[dict], dict]:
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
    return annotated_image, detections, counts


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
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"ok": False, "error": "Vui lòng chọn ảnh để dự đoán."}), 400

        confidence = float(request.form.get("conf", DEFAULT_CONFIDENCE))
        max_det = int(request.form.get("max_det", DEFAULT_MAX_DET))
        model_id = request.form.get("model_id", "finetune")

        image = Image.open(uploaded_file.stream).convert("RGB")
        model, selected_model = get_model(model_id)
        results = model.predict(
            source=np.array(image),
            conf=confidence,
            max_det=max_det,
            device="cpu",
            verbose=False,
        )
        annotated_image, detections, counts = parse_result(results[0])

        return jsonify(
            {
                "ok": True,
                "model_label": selected_model["label"],
                "original_image": image_to_base64(image),
                "annotated_image": image_to_base64(annotated_image),
                "detections": detections,
                "counts": counts,
                "summary": {
                    "total_detections": len(detections),
                    "classes_found": len(counts),
                },
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Lỗi dự đoán: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
