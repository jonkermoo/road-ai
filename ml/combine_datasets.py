"""
Combine police, pothole, and roadwork datasets into a single multi-class dataset.
This will create a new dataset structure in ml/train/combined/
"""

import os
import shutil
from pathlib import Path

# Source directories
POLICE_DIR = Path("train/police")
POTHOLE_DIR = Path("train/pothole")
ROADWORK_DIR = Path("train/roadwork")

# Output directory
COMBINED_DIR = Path("train/combined")

# Create output structure
for split in ["train", "valid", "test"]:
    (COMBINED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_and_relabel(source_dir: Path, class_id: int, split: str):
    """
    Copy images and labels from source dataset, updating class IDs.

    Args:
        source_dir: Source dataset directory (e.g., train/police)
        class_id: New class ID (0=police, 1=pothole, 2=roadwork)
        split: train/valid/test
    """
    src_images = source_dir / split / "images"
    src_labels = source_dir / split / "labels"

    dst_images = COMBINED_DIR / split / "images"
    dst_labels = COMBINED_DIR / split / "labels"

    if not src_images.exists():
        print(f"Warning: {src_images} doesn't exist, skipping...")
        return

    count = 0
    for img_file in src_images.glob("*.jpg"):
        # Copy image with prefix to avoid name collisions
        prefix = ["police", "pothole", "roadwork"][class_id]
        new_img_name = f"{prefix}_{img_file.name}"
        shutil.copy2(img_file, dst_images / new_img_name)

        # Copy and update label file
        label_file = src_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            new_label_name = f"{prefix}_{img_file.stem}.txt"

            # Read original labels and update class ID
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Update class IDs (first number in each line)
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    # Replace old class ID (0) with new class ID
                    parts[0] = str(class_id)
                    updated_lines.append(' '.join(parts) + '\n')

            # Write updated labels
            with open(dst_labels / new_label_name, 'w') as f:
                f.writelines(updated_lines)

        count += 1

    print(f"  Copied {count} {prefix} images from {split}")

# Combine all datasets
print("Combining datasets...")
print("\nProcessing police dataset (class 0)...")
for split in ["train", "valid", "test"]:
    copy_and_relabel(POLICE_DIR, 0, split)

print("\nProcessing pothole dataset (class 1)...")
for split in ["train", "valid", "test"]:
    copy_and_relabel(POTHOLE_DIR, 1, split)

print("\nProcessing roadwork dataset (class 2)...")
for split in ["train", "valid", "test"]:
    copy_and_relabel(ROADWORK_DIR, 2, split)

# Create combined data.yaml
yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 3
names: ['police', 'pothole', 'roadwork']
"""

with open(COMBINED_DIR / "data.yaml", 'w') as f:
    f.write(yaml_content)

print("\n✓ Dataset combination complete!")
print(f"✓ Combined dataset created at: {COMBINED_DIR}")
print(f"✓ data.yaml created with 3 classes: police, pothole, roadwork")

# Count final images
for split in ["train", "valid", "test"]:
    img_count = len(list((COMBINED_DIR / split / "images").glob("*.jpg")))
    print(f"  {split}: {img_count} images")
