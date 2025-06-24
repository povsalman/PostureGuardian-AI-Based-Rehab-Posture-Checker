import os
import random
import shutil
from pathlib import Path

# Paths
SOURCE_DIR = "dataset"
DEST_DIR = "dataset_small"

# Configuration
CLASS_NAMES = ["Arm Raise Correct", "Arm Raise Incorrect"]
SPLIT_RATIOS = {'train': 0.6, 'val': 0.2, 'test': 0.2}
SAMPLE_SIZE = 20  # Number of videos to randomly sample from each class

# Make folders
for split in SPLIT_RATIOS:
    for cls in CLASS_NAMES:
        Path(f"{DEST_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Process each class
for cls in CLASS_NAMES:
    video_files = os.listdir(os.path.join(SOURCE_DIR, cls))
    video_files = [f for f in video_files if f.endswith('.mp4')]  # filter only videos

    # Sample only a subset
    sampled_files = random.sample(video_files, SAMPLE_SIZE)

    # Shuffle and split
    random.shuffle(sampled_files)
    total = len(sampled_files)
    train_end = int(SPLIT_RATIOS['train'] * total)
    val_end = train_end + int(SPLIT_RATIOS['val'] * total)

    splits = {
        'train': sampled_files[:train_end],
        'val': sampled_files[train_end:val_end],
        'test': sampled_files[val_end:]
    }

    # Copy to destination
    for split, files in splits.items():
        for file in files:
            src_path = os.path.join(SOURCE_DIR, cls, file)
            dst_path = os.path.join(DEST_DIR, split, cls, file)
            shutil.copy2(src_path, dst_path)
