from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_label_dir(image_dir: Path) -> Path:
    # YOLO datasets thường có cấu trúc images/ và labels/ song song nhau.
    # Hàm này tìm đúng thư mục labels tương ứng với thư mục images hiện tại.
    candidate = image_dir.parent / "labels"
    if candidate.exists():
        return candidate
    alt = image_dir.parent.parent / "labels"
    return alt


def summarize_split(root: Path, split_name: str, image_rel: str, class_names: list[str]) -> dict:
    # Hàm này gom toàn bộ thống kê quan trọng của một split:
    # số ảnh, số label, số box, phân bố class, box lỗi, ảnh trùng base name...
    image_dir = root / image_rel
    label_dir = detect_label_dir(image_dir)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(p for p in label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    class_counts = Counter()
    area_sums = Counter()
    boxes_per_image: list[int] = []
    crowded_images: list[tuple[int, str]] = []
    duplicate_bases = Counter()
    bad_lines: list[str] = []

    # Đếm số lần lặp của cùng một ảnh gốc (base image) sau khi đã qua augment/export.
    for image_path in images:
        duplicate_bases[image_path.stem.split(".rf.")[0]] += 1

    # Đọc từng file nhãn YOLO và kiểm tra format từng dòng.
    for label_path in labels:
        text = label_path.read_text(encoding="utf-8").strip()
        box_count = 0
        if text:
            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    bad_lines.append(f"{label_path.name}:{line_number}:{line}")
                    continue
                try:
                    class_id = int(float(parts[0]))
                    _, _, width, height = map(float, parts[1:])
                except ValueError:
                    bad_lines.append(f"{label_path.name}:{line_number}:{line}")
                    continue
                if not 0 <= class_id < len(class_names):
                    bad_lines.append(f"{label_path.name}:{line_number}:invalid_class_{class_id}")
                    continue
                class_counts[class_id] += 1
                area_sums[class_id] += width * height
                box_count += 1
        boxes_per_image.append(box_count)
        crowded_images.append((box_count, label_path.name))

    crowded_images.sort(reverse=True)
    total_boxes = sum(class_counts.values())
    duplicate_groups = sum(1 for count in duplicate_bases.values() if count > 1)

    # Trả về một dict tổng hợp để cuối cùng xuất ra JSON dễ đọc/dễ phân tích.
    return {
        "split": split_name,
        "images": len(images),
        "labels": len(labels),
        "boxes_total": total_boxes,
        "avg_boxes_per_image": round(sum(boxes_per_image) / len(boxes_per_image), 2) if boxes_per_image else 0,
        "max_boxes_in_one_image": max(boxes_per_image) if boxes_per_image else 0,
        "missing_labels": sorted(image_stems - label_stems),
        "missing_images": sorted(label_stems - image_stems),
        "bad_lines": bad_lines,
        "class_counts": {class_names[i]: class_counts[i] for i in range(len(class_names))},
        "class_ratio_percent": {
            class_names[i]: round((class_counts[i] / total_boxes) * 100, 2) if total_boxes else 0
            for i in range(len(class_names))
        },
        "avg_box_area": {
            class_names[i]: round(area_sums[i] / class_counts[i], 5) if class_counts[i] else 0
            for i in range(len(class_names))
        },
        "top_crowded_images": crowded_images[:5],
        "unique_base_images": len(duplicate_bases),
        "duplicate_groups": duplicate_groups,
        "max_duplicate_count": max(duplicate_bases.values()) if duplicate_bases else 0,
        "top_duplicate_bases": [
            [base, count] for base, count in duplicate_bases.most_common(10) if count > 1
        ],
        "base_image_names": sorted(duplicate_bases.keys()),
    }


def main() -> None:
    # File này đang mặc định phân tích bộ dữ liệu gốc 5 class.
    dataset_root = Path("taco_merged_5class_yolo")
    data_yaml = dataset_root / "data.yaml"

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    class_names: list[str] = config["names"]

    split_map = {
        "train": config["train"],
        "val": config["val"],
        "test": config["test"],
    }

    report = {
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "splits": [],
    }

    overall_counts = Counter()
    for split_name, image_rel in split_map.items():
        split_report = summarize_split(dataset_root, split_name, image_rel, class_names)
        report["splits"].append(split_report)
        for class_name, count in split_report["class_counts"].items():
            overall_counts[class_name] += count

    # Kiểm tra rò rỉ dữ liệu giữa các split bằng cách so sánh base image.
    base_name_lookup = {
        split_report["split"]: set(split_report["base_image_names"])
        for split_report in report["splits"]
    }
    report["split_base_overlap"] = []
    for split_a, split_b in combinations(base_name_lookup, 2):
        overlap = sorted(base_name_lookup[split_a] & base_name_lookup[split_b])
        report["split_base_overlap"].append(
            {
                "pair": [split_a, split_b],
                "overlap_count": len(overlap),
                "sample_base_names": overlap[:20],
            }
        )

    for split_report in report["splits"]:
        del split_report["base_image_names"]

    report["overall_class_counts"] = dict(overall_counts)
    report["overall_boxes_total"] = sum(overall_counts.values())
    report["overall_class_ratio_percent"] = {
        name: round((overall_counts[name] / report["overall_boxes_total"]) * 100, 2)
        if report["overall_boxes_total"]
        else 0
        for name in class_names
    }

    # In JSON để tiện lưu log hoặc copy sang báo cáo/phân tích tiếp.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
