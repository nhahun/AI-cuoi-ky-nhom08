from __future__ import annotations

import shutil
from pathlib import Path

import yaml


"""
Gộp dataset gốc 5 class với dataset ngoài đã remap về 4 class.

Script này:
- copy ảnh và label từ hai nguồn dữ liệu,
- thêm prefix vào tên file để tránh trùng,
- tạo ra một dataset YOLO 5 class mới.

Sau bước này, dataset kết quả vẫn nên được resplit lại theo base image trước khi fine-tune.
"""

PRIMARY_ROOT = Path("taco_merged_5class_yolo")
SECONDARY_ROOT = Path("garbage_4class_yolo")
OUTPUT_ROOT = Path("merged_5class_yolo")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
TARGET_NAMES = ["plastic", "metal", "paper", "cardboard", "trash"]
PREFIXES = {
    "primary": "taco",
    "secondary": "garb",
}


def detect_label_dir(image_dir: Path) -> Path:
    # Tìm thư mục labels tương ứng với thư mục images của dataset hiện tại.
    candidate = image_dir.parent / "labels"
    if candidate.exists():
        return candidate
    alt = image_dir.parent.parent / "labels"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find labels directory for {image_dir}")


def dataset_split_dirs(root: Path, split_name: str) -> tuple[Path, Path]:
    # Hỗ trợ cả dataset đã chuẩn split/images lẫn dataset phải đọc qua data.yaml.
    images_dir = root / split_name / "images"
    if images_dir.exists():
        return images_dir, root / split_name / "labels"

    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    if split_name == "valid":
        image_rel = config["val"]
    else:
        image_rel = config[split_name]
    images_dir = root / image_rel
    labels_dir = detect_label_dir(images_dir)
    return images_dir, labels_dir


def copy_dataset(root: Path, split_name: str, prefix: str) -> tuple[int, int]:
    # Copy ảnh + nhãn sang dataset gộp và thêm prefix để tránh trùng tên file giữa hai bộ dữ liệu.
    source_images, source_labels = dataset_split_dirs(root, split_name)
    target_images = OUTPUT_ROOT / split_name / "images"
    target_labels = OUTPUT_ROOT / split_name / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    for image_path in sorted(source_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        new_stem = f"{prefix}_{image_path.stem}"
        shutil.copy2(image_path, target_images / f"{new_stem}{image_path.suffix.lower()}")
        shutil.copy2(label_path, target_labels / f"{new_stem}.txt")
        copied_images += 1
        copied_labels += 1

    return copied_images, copied_labels


def write_data_yaml() -> None:
    # Tạo data.yaml cho bộ dữ liệu 5 class sau khi gộp.
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_NAMES),
        "names": TARGET_NAMES,
    }
    (OUTPUT_ROOT / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    # Không ghi đè thư mục đầu ra để tránh mất dữ liệu đã gộp trước đó.
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_ROOT}. "
            "Remove it or rename OUTPUT_ROOT before running again."
        )

    summary: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split_name in SPLITS:
        primary_images, _ = copy_dataset(PRIMARY_ROOT, split_name, PREFIXES["primary"])
        secondary_images, _ = copy_dataset(SECONDARY_ROOT, split_name, PREFIXES["secondary"])
        summary[split_name]["primary_images"] = primary_images
        summary[split_name]["secondary_images"] = secondary_images
        summary[split_name]["total_images"] = primary_images + secondary_images

    write_data_yaml()

    # In tóm tắt để dễ kiểm tra số ảnh lấy từ từng nguồn.
    print("Merge complete")
    for split_name in SPLITS:
        split_summary = summary[split_name]
        print(
            f"- {split_name}: "
            f"primary={split_summary['primary_images']}, "
            f"secondary={split_summary['secondary_images']}, "
            f"total={split_summary['total_images']}"
        )


if __name__ == "__main__":
    main()
