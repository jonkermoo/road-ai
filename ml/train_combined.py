"""
Train a single YOLOv8 model on the combined dataset (police, pothole, roadwork).
This will be much faster than running 3 separate models!
"""

from ultralytics import YOLO
from pathlib import Path

# Configuration
DATA_YAML = Path("train/combined/data.yaml")
MODEL_SIZE = "yolov8n.pt"  # n=nano (fastest), s=small, m=medium, l=large, x=xlarge
EPOCHS = 100
IMGSZ = 640
BATCH = 16  # Adjust based on your GPU memory
PROJECT = "runs/detect"
NAME = "combined_road_ai"

def train():
    """Train the combined model."""

    print("=" * 60)
    print("Training Combined Road AI Model")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Classes: police, pothole, roadwork")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMGSZ}")
    print(f"Batch size: {BATCH}")
    print("=" * 60)

    # Load pretrained model
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=10,  # Early stopping
        save=True,
        device=0,  # Use GPU 0, or 'cpu' for CPU training
        workers=8,
        plots=True,
        val=True,
        cache=False,  # Set to True if you have enough RAM
    )

    print("\nTraining complete!")
    print(f"Best model saved at: {PROJECT}/{NAME}/weights/best.pt")
    print(f"Training results: {PROJECT}/{NAME}/")

    return results

if __name__ == "__main__":
    train()
