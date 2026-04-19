from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    # Cho phép thay đổi source/output/tỉ lệ split bằng dòng lệnh.
    parser = argparse.ArgumentParser(
        description="Resplit a YOLO dataset by base image name to avoid train/val/test leakage."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("taco_merged_5class_yolo"),
        help="Path to the source YOLO dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("taco_merged_5class_yolo_resplit"),
        help="Path to the output YOLO dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    # Tổng 3 tỉ lệ phải bằng 1.0, nếu không thì dừng sớm để tránh chia dữ liệu sai.
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")


def load_config(dataset_root: Path) -> dict:
    # Đọc file data.yaml của YOLO để biết đường dẫn ảnh và tên class.
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml in {dataset_root}")
    return yaml.safe_load(data_yaml.read_text(encoding="utf-8"))


def detect_label_dir(image_dir: Path) -> Path:
    # Hỗ trợ cả 2 kiểu bố cục dataset phổ biến: split/images và split/labels hoặc dạng tương đương.
    candidate = image_dir.parent / "labels"
    if candidate.exists():
        return candidate
    alt = image_dir.parent.parent / "labels"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find labels directory for {image_dir}")


def base_key(path: Path) -> str:
    # Chuẩn hóa "ảnh gốc" bằng cách bỏ phần .rf.... để các bản augment của cùng ảnh được gom về một nhóm.
    return path.stem.split(".rf.")[0].lower()


def collect_items(dataset_root: Path, config: dict) -> tuple[dict[str, list[dict]], list[str]]:
    # Gom toàn bộ ảnh/label theo base image để các bản sao augment không bị chia ra nhiều split khác nhau.
    grouped: dict[str, list[dict]] = defaultdict(list)
    class_names: list[str] = config["names"]

    for source_split, image_rel in (
        ("train", config["train"]),
        ("val", config["val"]),
        ("test", config["test"]),
    ):
        image_dir = dataset_root / image_rel
        label_dir = detect_label_dir(image_dir)

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for image {image_path}")
            grouped[base_key(image_path)].append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "source_split": source_split,
                }
            )

    return grouped, class_names


def assign_groups(
    grouped_items: dict[str, list[dict]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    # Ý tưởng chính:
    # - xáo trộn nhóm base image
    # - ưu tiên gán các nhóm lớn trước
    # - luôn gán vào split đang "thiếu" dữ liệu nhất so với tỉ lệ mục tiêu
    split_names = ("train", "valid", "test")
    split_ratios = {"train": train_ratio, "valid": val_ratio, "test": test_ratio}

    group_entries = [(group_name, items) for group_name, items in grouped_items.items()]
    rng = random.Random(seed)
    rng.shuffle(group_entries)
    group_entries.sort(key=lambda entry: len(entry[1]), reverse=True)

    total_images = sum(len(items) for _, items in group_entries)
    target_counts = {
        split_name: total_images * split_ratios[split_name] for split_name in split_names
    }

    assigned: dict[str, list[dict]] = {split_name: [] for split_name in split_names}
    current_counts = Counter()

    for _, items in group_entries:
        group_size = len(items)
        best_split = min(
            split_names,
            key=lambda split_name: (
                current_counts[split_name] / target_counts[split_name]
                if target_counts[split_name] > 0
                else float("inf"),
                current_counts[split_name],
            ),
        )
        assigned[best_split].extend(items)
        current_counts[best_split] += group_size

    return assigned


def copy_split(output_root: Path, split_name: str, items: list[dict]) -> None:
    # Tạo cấu trúc images/labels mới rồi copy file sang dataset đã resplit.
    images_dir = output_root / split_name / "images"
    labels_dir = output_root / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        shutil.copy2(item["image_path"], images_dir / item["image_path"].name)
        shutil.copy2(item["label_path"], labels_dir / item["label_path"].name)


def write_data_yaml(output_root: Path, class_names: list[str]) -> None:
    # Sinh lại data.yaml để dataset mới dùng được ngay với Ultralytics/YOLO.
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def summarize_assignment(assigned: dict[str, list[dict]]) -> None:
    # In nhanh ra console để kiểm tra split mới có cân đối tương đối hay không.
    print("Resplit summary")
    for split_name in ("train", "valid", "test"):
        items = assigned[split_name]
        unique_bases = {base_key(item["image_path"]) for item in items}
        source_counter = Counter(item["source_split"] for item in items)
        print(
            f"- {split_name}: {len(items)} images, {len(unique_bases)} base images, "
            f"from original splits {dict(source_counter)}"
        )


def main() -> None:
    # reconfigure giúp terminal Windows in tiếng Việt ổn hơn khi có thể.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}. "
            "Remove it or choose a different --output path."
        )

    config = load_config(source_root)
    grouped_items, class_names = collect_items(source_root, config)
    assigned = assign_groups(
        grouped_items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Sau khi chia xong thì ghi dataset mới ra ổ đĩa.
    output_root.mkdir(parents=True, exist_ok=False)
    for split_name in ("train", "valid", "test"):
        copy_split(output_root, split_name, assigned[split_name])
    write_data_yaml(output_root, class_names)
    summarize_assignment(assigned)
    print(f"New dataset written to: {args.output}")


if __name__ == "__main__":
    main()
