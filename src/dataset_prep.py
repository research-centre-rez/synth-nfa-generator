import os
import shutil
from pathlib import Path
import yaml
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

def prepare_yolo_dataset(
    raw_masks_dir: str | Path,
    raw_images_dir: str | Path,
    output_dataset_dir: str | Path,
    val_split: float = 0.2
) -> Path:
    """
    Converts binary segment masks to YOLO polygon labels and formats 
    the directory structure for Ultralytics training.
    """
    raw_masks_dir = Path(raw_masks_dir)
    raw_images_dir = Path(raw_images_dir)
    output_dataset_dir = Path(output_dataset_dir)

    # Temporary directory for generated labels
    temp_labels_dir = output_dataset_dir / "temp_labels"
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert BW masks to YOLO polygon txt files
    convert_segment_masks_to_yolo_seg(
        masks_dir=str(raw_masks_dir),
        output_dir=str(temp_labels_dir),
        classes=1
    )

    # 2. Setup standard YOLO dataset directory structure
    train_img_dir = output_dataset_dir / "train" / "images"
    train_lbl_dir = output_dataset_dir / "train" / "labels"
    val_img_dir = output_dataset_dir / "val" / "images"
    val_lbl_dir = output_dataset_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Pair and split images and converted labels
    image_files = sorted(list(raw_images_dir.glob("*.png")))
    num_val = max(1, int(len(image_files) * val_split)) if len(image_files) > 1 else 0

    for idx, img_path in enumerate(image_files):
        label_path = temp_labels_dir / f"{img_path.stem}.txt"
        
        # Decide split
        is_val = idx < num_val
        dest_img_dir = val_img_dir if is_val else train_img_dir
        dest_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        shutil.copy(img_path, dest_img_dir / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, dest_lbl_dir / label_path.name)

    # Clean up temp labels
    shutil.rmtree(temp_labels_dir)

    # 4. Generate dataset.yaml required by Ultralytics
    yaml_content = {
        "path": str(output_dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "anomaly"}
    }

    yaml_path = output_dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[Dataset Prep] YOLO dataset successfully prepared at: {output_dataset_dir}")
    return yaml_path