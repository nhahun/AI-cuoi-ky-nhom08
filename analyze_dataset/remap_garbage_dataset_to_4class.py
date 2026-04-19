from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import yaml


SOURCE_ROOT = Path("GARBAGE CLASSIFICATION")
OUTPUT_ROOT = Path("garbage_4class_yolo")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Source classes from data.yaml:
# 0 BIODEGRADABLE
# 1 CARDBOARD
# 2 GLASS
# 3 METAL
# 4 PAPER
# 5 PLASTIC
CLASS_MAP = {
    1: 3,  # CARDBOARD -> cardboard
    3: 1,  # METAL -> metal
    4: 2,  # PAPER -> paper
    5: 0,  # PLASTIC -> plastic
}
TARGET_NAMES = ["plastic", "metal", "paper", "cardboard"]


def rewrite_label_file(source_label: Path, target_label: Path, kept_counter: Counter, dropped_counter: Counter) -> bool:
    # Đọc từng file label YOLO và chỉ giữ các class nằm trong CLASS_MAP.
    # Nếu file không còn box nào hợp lệ sau khi lọc thì trả về False để bỏ luôn ảnh đó.
    output_lines: list[str] = []
    text = source_label.read_text(encoding="utf-8").strip()
    if not text:
        return False

    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        src_class = int(float(parts[0]))
        if src_class not in CLASS_MAP:
            dropped_counter[src_class] += 1
            continue
        dst_class = CLASS_MAP[src_class]
        kept_counter[dst_class] += 1
        output_lines.append(" ".join([str(dst_class), *parts[1:]]))

    if not output_lines:
        return False

    target_label.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return True


def process_split(split_name: str, kept_counter: Counter, dropped_counter: Counter) -> tuple[int, int]:
    # Xử lý riêng từng split để giữ nguyên cấu trúc train/valid/test của bộ dữ liệu nguồn.
    source_images = SOURCE_ROOT / split_name / "images"
    source_labels = SOURCE_ROOT / split_name / "labels"
    target_images = OUTPUT_ROOT / split_name / "images"
    target_labels = OUTPUT_ROOT / split_name / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    images_kept = 0
    images_dropped = 0

    # Chỉ copy ảnh nào còn ít nhất một label hợp lệ sau bước remap class.
    for image_path in sorted(source_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            images_dropped += 1
            continue

        target_label_path = target_labels / label_path.name
        has_kept_labels = rewrite_label_file(label_path, target_label_path, kept_counter, dropped_counter)
        if not has_kept_labels:
            images_dropped += 1
            if target_label_path.exists():
                target_label_path.unlink()
            continue

        shutil.copy2(image_path, target_images / image_path.name)
        images_kept += 1

    return images_kept, images_dropped


def write_data_yaml() -> None:
    # Tạo file data.yaml mới cho bộ dữ liệu 4 class sau khi remap.
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
    # Không ghi đè dataset cũ để tránh mất dữ liệu gốc.
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_ROOT}. "
            "Remove it or rename OUTPUT_ROOT before running again."
        )

    kept_counter: Counter = Counter()
    dropped_counter: Counter = Counter()
    split_summary = {}

    for split_name in ("train", "valid", "test"):
        kept, dropped = process_split(split_name, kept_counter, dropped_counter)
        split_summary[split_name] = {"images_kept": kept, "images_dropped": dropped}

    write_data_yaml()

    # In thống kê để biết remap đã giữ lại được bao nhiêu dữ liệu và bỏ đi bao nhiêu box.
    print("Remap complete")
    for split_name, stats in split_summary.items():
        print(
            f"- {split_name}: kept {stats['images_kept']} images, "
            f"dropped {stats['images_dropped']} images"
        )

    print("Kept boxes by target class:")
    for idx, name in enumerate(TARGET_NAMES):
        print(f"- {name}: {kept_counter[idx]}")

    source_names = {
        0: "BIODEGRADABLE",
        1: "CARDBOARD",
        2: "GLASS",
        3: "METAL",
        4: "PAPER",
        5: "PLASTIC",
    }
    print("Dropped boxes by source class:")
    for src_idx in sorted(dropped_counter):
        print(f"- {source_names[src_idx]}: {dropped_counter[src_idx]}")


if __name__ == "__main__":
    main()
