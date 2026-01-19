#!/usr/bin/env python3
# prep_fastmfcc.py - Extract MFCC features from processed audio files

import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from glob import glob
from collections import defaultdict

# ========= CONFIG =========
PROCESSED_AUDIO_ROOT = Path(r"C:\Users\jason\Documents\Depthwise Separable CNN Audio Processing\Split_Audio")
FEAT_OUTPUT_ROOT = Path(r"C:\Users\jason\Documents\Depthwise Separable CNN Audio Processing\KWS_MFCC32_UltraTiny")

# MFCC parameters
SR = 16000
N_FFT = 512
HOP_LENGTH = 160  # ~10ms hop
N_MELS = 32
N_MFCC = 13
FMIN = 20
FMAX = SR // 2

# Expected audio length (1 second at 16kHz = 16000 samples)
EXPECTED_SAMPLES = 16000

# Target categories
TARGET_CATEGORIES = ["on", "off", "_background_noise_", "unknown"]

print("MFCC Feature Extractor")
print(f"Input:  {PROCESSED_AUDIO_ROOT}")
print(f"Output: {FEAT_OUTPUT_ROOT}")
print(f"Target categories: {TARGET_CATEGORIES}")
print(f"MFCC config: n_mfcc={N_MFCC}, sr={SR}, hop={HOP_LENGTH}")
print()


def extract_mfcc_features(audio_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio
        y, sr = sf.read(audio_path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)  # Convert to mono

        # Ensure correct sample rate
        if sr != SR:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=SR)

        # Ensure correct length (should already be 1 second from processing)
        if len(y) != EXPECTED_SAMPLES:
            if len(y) < EXPECTED_SAMPLES:
                y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)))
            else:
                y = y[:EXPECTED_SAMPLES]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            center=True,
            pad_mode='constant'
        )

        # Transpose to (time, mfcc) format
        mfcc = mfcc.T  # (time_frames, n_mfcc)

        # Ensure consistent time dimension (should be ~66 frames for 1 second)
        expected_frames = 1 + (EXPECTED_SAMPLES - N_FFT) // HOP_LENGTH
        if mfcc.shape[0] != expected_frames:
            if mfcc.shape[0] < expected_frames:
                # Pad with last frame
                pad_frames = expected_frames - mfcc.shape[0]
                last_frame = mfcc[-1:] if mfcc.shape[0] > 0 else np.zeros((1, N_MFCC))
                padding = np.repeat(last_frame, pad_frames, axis=0)
                mfcc = np.vstack([mfcc, padding])
            else:
                # Trim to expected frames
                mfcc = mfcc[:expected_frames]

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"    ERROR processing {audio_path}: {e}")
        return None


def process_split(split_name):
    """Process one split (train/val/test)"""
    print(f"Processing {split_name} split...")

    input_split_dir = PROCESSED_AUDIO_ROOT / split_name
    output_split_dir = FEAT_OUTPUT_ROOT / split_name

    if not input_split_dir.exists():
        print(f"  WARNING: {input_split_dir} does not exist, skipping")
        return []

    # Create output directories for each category
    for category in TARGET_CATEGORIES:
        (output_split_dir / category).mkdir(parents=True, exist_ok=True)

    all_features = []
    stats = defaultdict(int)

    # Process each category directory
    for category in TARGET_CATEGORIES:
        input_category_dir = input_split_dir / category
        output_category_dir = output_split_dir / category

        if not input_category_dir.exists():
            print(f"    WARNING: {input_category_dir} not found, skipping")
            continue


        wav_files = list(glob(str(input_category_dir / "*.wav")))
        print(f"    [DEBUG] {category} in {split_name}: found {len(wav_files)} wav files in {input_category_dir}")
        if not wav_files:
            print(f"    WARNING: No .wav files in {input_category_dir}")
            continue

        print(f"  Processing {category}: {len(wav_files)} files")

        processed = 0
        for wav_file in sorted(wav_files):
            mfcc = extract_mfcc_features(wav_file)
            if mfcc is not None:
                # Save feature file
                base_name = Path(wav_file).stem
                output_path = output_category_dir / f"{base_name}.npy"
                np.save(output_path, mfcc)

                all_features.append(mfcc)
                processed += 1

            if processed % 100 == 0 and processed > 0:
                print(f"    Processed {processed}/{len(wav_files)} files...")

        stats[category] = processed
        print(f"    ✓ {category}: {processed} features saved")

    print(f"  {split_name} summary: {dict(stats)}")
    print(f"  Total features extracted: {sum(stats.values())}")

    return all_features


def compute_cmvn_stats(all_train_features):
    """Compute Channel-wise Mean and Variance Normalization statistics"""
    print("\nComputing CMVN statistics from training data...")

    if not all_train_features:
        print("ERROR: No training features available for CMVN computation")
        return

    # Stack all training features
    all_mfcc = np.vstack(all_train_features)  # (total_frames, n_mfcc)
    print(f"CMVN computation: {all_mfcc.shape[0]:,} frames × {all_mfcc.shape[1]} MFCC coefficients")

    # Compute mean and std across all time frames
    mean = np.mean(all_mfcc, axis=0, dtype=np.float32)
    std = np.std(all_mfcc, axis=0, dtype=np.float32)

    # Avoid division by zero
    std = np.maximum(std, 1e-8)

    print(f"MFCC statistics:")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range:  [{std.min():.3f}, {std.max():.3f}]")

    # Save CMVN statistics
    cmvn_path = FEAT_OUTPUT_ROOT / "cmvn_mfcc_train.npz"
    np.savez(cmvn_path, mean=mean, std=std)
    print(f"✓ CMVN stats saved: {cmvn_path}")

    return mean, std


def validate_features(split_name):
    """Validate extracted features"""
    split_dir = FEAT_OUTPUT_ROOT / split_name
    if not split_dir.exists():
        return

    print(f"\nValidating {split_name} features...")

    for category in TARGET_CATEGORIES:
        category_dir = split_dir / category
        if category_dir.exists():
            npy_files = list(category_dir.glob("*.npy"))
            if npy_files:
                # Check first file
                sample_feat = np.load(npy_files[0])
                print(f"  {category}: {len(npy_files)} files, shape={sample_feat.shape}")
            else:
                print(f"  {category}: 0 files")


def main():
    """Main feature extraction pipeline"""

    # Validate input directory
    if not PROCESSED_AUDIO_ROOT.exists():
        print(f"ERROR: Input directory does not exist: {PROCESSED_AUDIO_ROOT}")
        print("Make sure you've run the audio processing script first!")
        return

    # Create output root
    FEAT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Find available splits
    splits = []
    for split_name in ["train", "val", "test"]:
        if (PROCESSED_AUDIO_ROOT / split_name).exists():
            splits.append(split_name)

    if not splits:
        print("ERROR: No train/val/test directories found!")
        return

    print(f"Found splits: {', '.join(splits)}")
    print()

    # Process each split
    all_train_features = []

    for split_name in splits:
        features = process_split(split_name)
        if split_name == "train":
            all_train_features = features

    # Compute CMVN stats from training data
    if all_train_features:
        compute_cmvn_stats(all_train_features)
    else:
        print("WARNING: No training features found, cannot compute CMVN stats")

    # Validate all splits
    for split_name in splits:
        validate_features(split_name)

    print(f"\n{'=' * 60}")
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Features saved to: {FEAT_OUTPUT_ROOT}")
    print("Ready for training!")


if __name__ == "__main__":
    main()