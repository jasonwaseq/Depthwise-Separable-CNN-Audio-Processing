import os
from glob import glob
import numpy as np
import soundfile as sf
import librosa
from collections import defaultdict

# ========= CONFIG =========
# Input from your split data
input_root = r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Split_Audio"  # Output from your splitter
output_root = r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Processed_Audio"  # Final processed data

# Target categories (everything else goes to unknown)
TARGET_CATEGORIES = {"on", "off", "_background_noise_"}

TARGET_SR = 16000
TARGET_SAMPLES = 16000
PEAK_HEADROOM = 0.999

# --- augment configs ---
# for on/off/unknown
APPLY_JITTER = True
JITTER_DB_RANGE = (-6.0, 6.0)
COPIES_JITTER = 1

APPLY_NOISE_MIX = True
SNR_DB_CHOICES = [20, 10, 5, 0]  # dB
COPIES_NOISE = 1

APPLY_BOTH = True
BOTH_JITTER_RANGE = (-3.0, 3.0)
BOTH_SNR_CHOICES = [10, 5]
COPIES_BOTH = 1

# for background_noise
APPLY_BG_JITTER = True
BG_JITTER_DB_RANGE = (-6.0, 6.0)
COPIES_BG_JITTER = 1

rng = np.random.default_rng(0)  # reproducible


# ========= UTILS =========
def load_audio_file(file_path):
    """Load audio file and convert to mono if needed"""
    try:
        y, sr = sf.read(file_path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32, copy=False), sr
    except Exception as e:
        print(f"    ERROR loading {file_path}: {e}")
        return None, None


def fix_sampling_rate(y, sr, target_sr=TARGET_SR):
    """Resample audio to target sample rate"""
    if sr == target_sr:
        return y, False
    z = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
    return z.astype(np.float32, copy=False), True


def fix_length_1s(y, n=TARGET_SAMPLES):
    """Pad or trim audio to exactly 1 second"""
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
        status = "padded"
    elif len(y) > n:
        y = y[:n]
        status = "trimmed"
    else:
        status = "ok"
    return y, status


def peak_cap(y, headroom=PEAK_HEADROOM):
    """Apply peak limiting to prevent clipping"""
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak > headroom and peak > 0:
        y = y * (headroom / peak)
    return y.astype(np.float32, copy=False)


def apply_loudness_jitter(y, db_range, _rng=rng):
    """Apply random loudness variation"""
    if not db_range or (db_range[0] == 0 and db_range[1] == 0):
        return y, 0.0
    delta_db = float(_rng.uniform(db_range[0], db_range[1]))
    g = 10.0 ** (delta_db / 20.0)
    z = y * g
    return peak_cap(z, PEAK_HEADROOM), delta_db


def rms(x):
    """Calculate RMS value"""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def choose_noise_segment(noise_path, length_samples, target_sr=TARGET_SR):
    """Extract random segment from noise file"""
    y, sr = load_audio_file(noise_path)
    if y is None:
        return np.zeros(length_samples, dtype=np.float32)

    y, _ = fix_sampling_rate(y, sr, target_sr)
    if len(y) < length_samples:
        reps = int(np.ceil(length_samples / max(1, len(y))))
        y = np.tile(y, reps)
    start_max = max(0, len(y) - length_samples)
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    return y[start:start + length_samples]


def mix_at_snr(clean, noise, snr_db):
    """Mix clean signal with noise at specified SNR"""
    rc = rms(clean)
    rn = rms(noise)
    if rc == 0.0 or rn == 0.0:
        return peak_cap(clean, PEAK_HEADROOM)
    target_rn = rc / (10.0 ** (snr_db / 20.0))
    scale = target_rn / rn
    mixed = clean + noise * scale
    return peak_cap(mixed, PEAK_HEADROOM)


def ensure_output_dirs(out_split_dir):
    """Create the four target category directories"""
    for category in ("background_noise", "on", "off", "unknown"):
        os.makedirs(os.path.join(out_split_dir, category), exist_ok=True)


def get_noise_files(split_dir):
    """Get all noise files from background_noise directory in this split"""
    bg_dir = os.path.join(split_dir, "background_noise")
    return sorted(glob(os.path.join(bg_dir, "*.wav"))) if os.path.isdir(bg_dir) else []


def categorize_class(class_name):
    """Map class names to target categories"""
    class_lower = class_name.lower()
    if class_lower == "on":
        return "on"
    elif class_lower == "off":
        return "off"
    elif class_lower == "background_noise":
        return "background_noise"
    else:
        return "unknown"  # Everything else goes to unknown


def process_audio_file(file_path, output_dir, target_category, noise_files, original_class=None):
    """Process a single audio file with augmentations"""
    y, sr = load_audio_file(file_path)
    if y is None:
        return False

    y, _ = fix_sampling_rate(y, sr, TARGET_SR)
    y, _ = fix_length_1s(y, TARGET_SAMPLES)
    y = peak_cap(y, PEAK_HEADROOM)

    base = os.path.splitext(os.path.basename(file_path))[0]

    # Add original class prefix for unknown category
    if target_category == "unknown" and original_class:
        base = f"{original_class}__{base}"

    # 0) Save clean version
    sf.write(os.path.join(output_dir, f"{base}.wav"), y, TARGET_SR, subtype="PCM_16")
    files_created = 1

    # Apply different augmentations based on category
    if target_category == "background_noise":
        # Background noise gets jitter only
        if APPLY_BG_JITTER and COPIES_BG_JITTER > 0:
            for i in range(COPIES_BG_JITTER):
                yj, ddb = apply_loudness_jitter(y, BG_JITTER_DB_RANGE, _rng=rng)
                suffix = f"_jit_{ddb:+.1f}dB".replace("+", "p").replace("-", "m")
                sf.write(os.path.join(output_dir, f"{base}{suffix}.wav"), yj, TARGET_SR, subtype="PCM_16")
                files_created += 1

    else:  # on, off, unknown get full augmentation
        # 1) Jitter-only
        if APPLY_JITTER and COPIES_JITTER > 0:
            for i in range(COPIES_JITTER):
                yj, ddb = apply_loudness_jitter(y, JITTER_DB_RANGE, _rng=rng)
                suffix = f"_jit_{ddb:+.1f}dB".replace("+", "p").replace("-", "m")
                sf.write(os.path.join(output_dir, f"{base}{suffix}.wav"), yj, TARGET_SR, subtype="PCM_16")
                files_created += 1

        # 2) Noise-only
        if APPLY_NOISE_MIX and COPIES_NOISE > 0 and noise_files:
            for i in range(COPIES_NOISE):
                snr = int(rng.choice(SNR_DB_CHOICES))
                nz = choose_noise_segment(rng.choice(noise_files), TARGET_SAMPLES, TARGET_SR)
                y_nm = mix_at_snr(y, nz, snr)
                sf.write(os.path.join(output_dir, f"{base}_nm_snr{snr}dB.wav"), y_nm, TARGET_SR, subtype="PCM_16")
                files_created += 1

        # 3) Both jitter and noise
        if APPLY_BOTH and COPIES_BOTH > 0 and noise_files:
            for i in range(COPIES_BOTH):
                yj_small, ddb = apply_loudness_jitter(y, BOTH_JITTER_RANGE, _rng=rng)
                snr = int(rng.choice(BOTH_SNR_CHOICES))
                nz = choose_noise_segment(rng.choice(noise_files), TARGET_SAMPLES, TARGET_SR)
                y_both = mix_at_snr(yj_small, nz, snr)
                suffix = f"_both_snr{snr}dB_j{ddb:+.1f}dB".replace("+", "p").replace("-", "m")
                sf.write(os.path.join(output_dir, f"{base}{suffix}.wav"), y_both, TARGET_SR, subtype="PCM_16")
                files_created += 1

    return files_created


def process_split(split_name):
    """Process one split (train, val, or test)"""
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'=' * 60}")

    in_split_dir = os.path.join(input_root, split_name)
    out_split_dir = os.path.join(output_root, split_name)

    if not os.path.exists(in_split_dir):
        print(f"WARNING: Split directory not found: {in_split_dir}")
        return

    # Create output directories
    ensure_output_dirs(out_split_dir)

    # Get noise files for this split (for augmentation)
    noise_files = get_noise_files(in_split_dir)
    if not noise_files and (APPLY_NOISE_MIX or APPLY_BOTH):
        print(f"WARNING: No background_noise files found for {split_name} split")

    # Statistics tracking
    stats = defaultdict(lambda: defaultdict(int))

    # Process each class directory in this split
    class_dirs = [d for d in os.listdir(in_split_dir)
                  if os.path.isdir(os.path.join(in_split_dir, d)) and not d.startswith(".")]

    for class_name in sorted(class_dirs):
        class_dir = os.path.join(in_split_dir, class_name)
        target_category = categorize_class(class_name)
        output_dir = os.path.join(out_split_dir, target_category)

        print(f"  Processing {class_name} -> {target_category}")

        wav_files = glob(os.path.join(class_dir, "*.wav"))
        if not wav_files:
            print(f"    WARNING: No .wav files found in {class_dir}")
            continue

        files_processed = 0
        total_files_created = 0

        for wav_file in wav_files:
            files_created = process_audio_file(
                wav_file, output_dir, target_category, noise_files, class_name
            )
            if files_created > 0:
                files_processed += 1
                total_files_created += files_created

        stats[target_category]["original_files"] += files_processed
        stats[target_category]["total_output_files"] += total_files_created

        print(f"    ✓ {files_processed} files processed → {total_files_created} output files")

    # Print split statistics
    print(f"\n{split_name.upper()} SPLIT SUMMARY:")
    print("-" * 50)
    for category in ("on", "off", "background_noise", "unknown"):
        orig = stats[category]["original_files"]
        total = stats[category]["total_output_files"]
        if orig > 0:
            multiplier = total / orig if orig > 0 else 0
            print(f"{category:<18}: {orig:>4} → {total:>5} files ({multiplier:.1f}x)")
        else:
            print(f"{category:<18}: {orig:>4} → {total:>5} files")


# ========= MAIN EXECUTION =========
def main():
    """Main processing function"""
    print("Audio Dataset Processor")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Target categories: {sorted(TARGET_CATEGORIES | {'unknown'})}")
    print()

    # Validate input directory
    if not os.path.exists(input_root):
        print(f"ERROR: Input directory does not exist: {input_root}")
        return

    # Create output root
    os.makedirs(output_root, exist_ok=True)

    # Find available splits
    splits = []
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(input_root, split_name)
        if os.path.isdir(split_path):
            splits.append(split_name)

    if not splits:
        print("ERROR: No train/val/test directories found in input directory")
        return

    print(f"Found splits: {', '.join(splits)}")

    # Process each split
    overall_stats = defaultdict(lambda: defaultdict(int))

    for split_name in splits:
        process_split(split_name)

        # Accumulate overall stats
        out_split_dir = os.path.join(output_root, split_name)
        for category in ("on", "off", "background_noise", "unknown"):
            category_dir = os.path.join(out_split_dir, category)
            if os.path.exists(category_dir):
                file_count = len(glob(os.path.join(category_dir, "*.wav")))
                overall_stats[split_name][category] = file_count

    # Print overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL DATASET SUMMARY")
    print(f"{'=' * 80}")

    print(f"{'Category':<18} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)

    category_totals = defaultdict(int)
    split_totals = defaultdict(int)

    for category in ("on", "off", "background_noise", "unknown"):
        train_count = overall_stats["train"][category]
        val_count = overall_stats["val"][category]
        test_count = overall_stats["test"][category]
        total_count = train_count + val_count + test_count

        print(f"{category:<18} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")

        category_totals[category] = total_count
        split_totals["train"] += train_count
        split_totals["val"] += val_count
        split_totals["test"] += test_count

    print("-" * 60)
    total_files = sum(category_totals.values())
    print(
        f"{'TOTAL':<18} {split_totals['train']:<8} {split_totals['val']:<8} {split_totals['test']:<8} {total_files:<8}")

    if total_files > 0:
        print(f"\nSplit percentages:")
        print(f"Train: {split_totals['train'] / total_files * 100:.1f}%")
        print(f"Val:   {split_totals['val'] / total_files * 100:.1f}%")
        print(f"Test:  {split_totals['test'] / total_files * 100:.1f}%")

    print(f"\nProcessed dataset saved to: {output_root}")


if __name__ == "__main__":
    main()