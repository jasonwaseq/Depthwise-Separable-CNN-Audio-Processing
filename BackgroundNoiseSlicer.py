import soundfile as sf
import librosa
import numpy as np
import os
from glob import glob

input_file_path = "/Users/ryanhoang/Desktop/KWS/Raw_Audio"
output_file_path = "/Users/ryanhoang/Desktop/KWS/Processed_Audio"

NOISE_INPUT_DIR = r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\_background_noise_"
SPLIT_DIRS = {
    'train': r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Split_Audio\train\_background_noise_",
    'val': r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Split_Audio\val\_background_noise_",
    'test': r"C:\Users\jason\OneDrive\Documents\Depthwise Separable CNN Audio Processing\Split_Audio\test\_background_noise_"
}
TARGET_SR = 16000
TARGET_SAMPLES = 16000
HOP_SAMPLES = 1000
PEAK_HEADROOM = 0.9999


def load_audio_file(file_path):
    """Load audio file as mono float32."""
    data, sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:  # mix down to mono
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)
    return data, sr


def fix_sampling_rate(y, sr, target_sr=TARGET_SR):
    """Resample only if needed. Returns (y, did_resample)."""
    if sr == target_sr:
        return y, False
    y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
    return y, True


def peak_cap(y, headroom=PEAK_HEADROOM):
    """Scale down if peak exceeds headroom."""
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak > headroom and peak > 0:
        y = y * (headroom / peak)
    return y.astype(np.float32, copy=False)


def slice_background_noise():
    for split, out_dir in SPLIT_DIRS.items():
        os.makedirs(out_dir, exist_ok=True)

    total_created = {split: 0 for split in SPLIT_DIRS}
    for wav_path in glob(os.path.join(NOISE_INPUT_DIR, "*.wav")):
        try:
            y, sr = load_audio_file(wav_path)
            if y is None or y.size == 0:
                print(f"[WARN] Empty audio: {wav_path}")
                continue
            y, _ = fix_sampling_rate(y, sr, TARGET_SR)
            n = len(y)
            base = os.path.splitext(os.path.basename(wav_path))[0]
            # Skip files shorter than one full slice
            last_valid_start = n - TARGET_SAMPLES
            if last_valid_start < 0:
                print(f"{os.path.basename(wav_path)} -> 0 noise slices (too short)")
                continue
            # Fixed-hop slicing (ditches any short tail)
            slices = []
            for start in range(0, last_valid_start + 1, HOP_SAMPLES):
                seg = y[start:start + TARGET_SAMPLES]
                seg = peak_cap(seg, PEAK_HEADROOM)
                slices.append(seg)
            # Split slices into train/val/test (80/10/10)
            n_total = len(slices)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            n_test = n_total - n_train - n_val
            split_counts = {'train': n_train, 'val': n_val, 'test': n_test}
            idx = 0
            for split in ['train', 'val', 'test']:
                out_dir = SPLIT_DIRS[split]
                for i in range(split_counts[split]):
                    seg = slices[idx]
                    out_name = f"{base}_seg{idx:05d}_start{idx*HOP_SAMPLES}.wav"
                    out_path = os.path.join(out_dir, out_name)
                    sf.write(out_path, seg, TARGET_SR, subtype="PCM_16")
                    total_created[split] += 1
                    idx += 1
            print(f"{os.path.basename(wav_path)} -> {n_total} noise slices (train: {split_counts['train']}, val: {split_counts['val']}, test: {split_counts['test']})")
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")
    print(f"Background noise slices created: {total_created}")
    return total_created


if __name__ == "__main__":
    slice_background_noise()
