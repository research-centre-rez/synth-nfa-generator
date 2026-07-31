import argparse
from pathlib import Path
from ultralytics import YOLO
from src.dataset_prep import prepare_yolo_dataset

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train YOLO Segmentation Pipeline for Anomaly Detection")
#     parser.add_argument("--raw-images", type=str, default="data/raw/images", help="Path to raw image directory")
#     parser.add_argument("--raw-masks", type=str, default="data/raw/masks", help="Path to raw BW masks directory")
#     parser.add_argument("--output-dataset", type=str, default="data/yolo_dataset", help="Path for YOLO dataset")
#     parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Pretrained YOLO seg model weights")
#     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
#     parser.add_argument("--imgsz", type=int, default=640, help="Image size")
#     return parser.parse_args()

def main():
    # args = parse_args()

    # Step 1: Dataset Conversion & Formatting
    print(">>> Preparing Dataset...")
    yaml_config_path = prepare_yolo_dataset(
        raw_masks_dir="data/raw/masks",
        raw_images_dir="data/raw/images",
        output_dataset_dir="data/yolo_dataset"
    )

    # Step 2: Initialize YOLO Segmentation Model
    # print(f">>> Loading Model: {args.model}")
    model = YOLO("yolo26n.pt")

    # Step 3: Train
    print(">>> Starting Training...")
    results = model.train(
        data=str(yaml_config_path),
        epochs=1,
        imgsz=(900, 1600),
        project="runs/anomaly_seg",
        name="demo_run",
        exist_ok=True
    )
    
    print(">>> Training complete. Model saved in runs/anomaly_seg/demo_run")

if __name__ == "__main__":
    main()