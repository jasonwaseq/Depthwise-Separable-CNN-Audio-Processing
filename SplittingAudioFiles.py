import os
import shutil
import random
from glob import glob
from collections import defaultdict

# Paths
RAW_DIR = r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing"
OUT_DIR = r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Split_Audio"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Minimum files per class to proceed with split
MIN_FILES_PER_CLASS = 10

random.seed(42)  # reproducibility


def validate_split_ratios(train_ratio, val_ratio, test_ratio):
    """Validate that split ratios sum to 1.0"""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def split_and_copy(class_dir, out_dir, train_ratio, val_ratio, test_ratio, min_files=10):
    """
    Split audio files from a class directory into train/val/test sets

    Returns:
        dict: Statistics about the split (train_count, val_count, test_count)
    """
    try:
        wavs = glob(os.path.join(class_dir, "*.wav"))

        if len(wavs) == 0:
            print(f"  WARNING: No .wav files found in {class_dir}")
            return {"train_count": 0, "val_count": 0, "test_count": 0, "skipped": True}

        if len(wavs) < min_files:
            print(f"  WARNING: Only {len(wavs)} files found (minimum {min_files} required). Skipping class.")
            return {"train_count": 0, "val_count": 0, "test_count": 0, "skipped": True}

        random.shuffle(wavs)

        n_total = len(wavs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Ensure all files are assigned

        splits = {
            "train": wavs[:n_train],
            "val": wavs[n_train:n_train + n_val],
            "test": wavs[n_train + n_val:],
        }

        class_name = os.path.basename(class_dir)
        stats = {}

        for split, files in splits.items():
            split_dir = os.path.join(out_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for f in files:
                try:
                    shutil.copy2(f, split_dir)
                except Exception as e:
                    print(f"  ERROR: Failed to copy {f} to {split_dir}: {e}")
                    return {"train_count": 0, "val_count": 0, "test_count": 0, "skipped": True}

            stats[f"{split}_count"] = len(files)

        stats["skipped"] = False
        return stats

    except Exception as e:
        print(f"  ERROR: Failed to process {class_dir}: {e}")
        return {"train_count": 0, "val_count": 0, "test_count": 0, "skipped": True}


def print_statistics(all_stats):
    """Print comprehensive statistics about the dataset split"""
    print("\n" + "=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)

    total_train = sum(stats["train_count"] for stats in all_stats.values() if not stats["skipped"])
    total_val = sum(stats["val_count"] for stats in all_stats.values() if not stats["skipped"])
    total_test = sum(stats["test_count"] for stats in all_stats.values() if not stats["skipped"])
    total_files = total_train + total_val + total_test

    processed_classes = sum(1 for stats in all_stats.values() if not stats["skipped"])
    skipped_classes = sum(1 for stats in all_stats.values() if stats["skipped"])

    print(f"Classes processed: {processed_classes}")
    print(f"Classes skipped: {skipped_classes}")
    print(f"Total files processed: {total_files}")
    print()

    if total_files > 0:
        print(f"Train: {total_train:,} files ({total_train / total_files * 100:.1f}%)")
        print(f"Val:   {total_val:,} files ({total_val / total_files * 100:.1f}%)")
        print(f"Test:  {total_test:,} files ({total_test / total_files * 100:.1f}%)")
        print()

    print("Per-class breakdown:")
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Status'}")
    print("-" * 65)

    for class_name, stats in sorted(all_stats.items()):
        if stats["skipped"]:
            print(f"{class_name:<20} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'SKIPPED'}")
        else:
            class_total = stats["train_count"] + stats["val_count"] + stats["test_count"]
            print(f"{class_name:<20} {stats['train_count']:<8} {stats['val_count']:<8} "
                  f"{stats['test_count']:<8} {class_total:<8} {'OK'}")


def main():
    """Main function to orchestrate the dataset splitting process"""
    print("Audio Dataset Splitter")
    print(f"Source: {RAW_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Minimum files per class: {MIN_FILES_PER_CLASS}")
    print()

    # Validate inputs
    try:
        validate_split_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: Source directory does not exist: {RAW_DIR}")
        return

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Process each class directory
    all_stats = {}
    class_dirs = [d for d in os.listdir(RAW_DIR)
                  if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith(".")]

    if not class_dirs:
        print(f"ERROR: No class directories found in {RAW_DIR}")
        return

    print(f"Found {len(class_dirs)} class directories to process")
    print()

    for class_name in sorted(class_dirs):
        class_path = os.path.join(RAW_DIR, class_name)
        print(f"Processing class: {class_name}")

        stats = split_and_copy(
            class_path, OUT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, MIN_FILES_PER_CLASS
        )
        all_stats[class_name] = stats

        if not stats["skipped"]:
            print(f"  ✓ Split: {stats['train_count']} train, {stats['val_count']} val, {stats['test_count']} test")
        print()

    # Print final statistics
    print_statistics(all_stats)
    print(f"\nTrain/Val/Test split completed!")
    print(f"Output saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()