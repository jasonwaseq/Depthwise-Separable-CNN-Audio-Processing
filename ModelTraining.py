#!/usr/bin/env python3
# Training.py — UltraTiny-DSCNN KWS with 4 classes + full INT8 PTQ export

import os, random, tempfile, shutil
from pathlib import Path
import numpy as np

# ========= EDIT THESE PATHS =========
FEAT_ROOT = Path(r"C:\Users\jason\Documents\Depthwise Separable CNN Audio Processing\KWS_MFCC32_UltraTiny")
OUT_DIR = Path(r"C:\Users\jason\Documents\Depthwise Separable CNN Audio Processing\artifacts_kws_ultratiny_int8")
# ====================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

import tensorflow as tf

keras = tf.keras
from tensorflow.keras import layers

# ---------- Constants ----------
CLASSES = ["on", "off", "_background_noise_", "unknown"]  # 4 classes for complete KWS
N_CLASSES = len(CLASSES)
N_MFCC = 13
INPUT_SHAPE = (66, 1, 13)  # NHWC (time, 1, mfcc)

print(f"Training model with {N_CLASSES} classes: {CLASSES}")


# ---------- Load CMVN ----------
def load_cmvn(feat_root: Path):
    """Load channel-wise mean and variance normalization parameters"""
    cmvn_path = feat_root / "cmvn_mfcc_train.npz"
    if not cmvn_path.exists():
        print(f"ERROR: CMVN file not found at {cmvn_path}")
        print("You need to run feature extraction first to generate MFCC features and CMVN stats")
        raise FileNotFoundError(f"CMVN file missing: {cmvn_path}")

    z = np.load(cmvn_path)
    mean = z["mean"].astype(np.float32)
    std = z["std"].astype(np.float32)
    print(f"Loaded CMVN: mean shape={mean.shape}, std shape={std.shape}")
    return mean, std


try:
    CMVN_MEAN, CMVN_STD = load_cmvn(FEAT_ROOT)
    CMVN_VAR = CMVN_STD ** 2
except FileNotFoundError:
    print("Please run feature extraction (prep_fastmfcc.py) on your Processed_Audio first!")
    exit(1)


# ---------- Dataset I/O ----------
def scan_split(root: Path):
    """Scan for .npy feature files and extract labels"""
    paths = sorted([p for p in root.rglob("*.npy")])
    if not paths:
        print(f"ERROR: No .npy files found under {root}")
        print("Make sure you've run feature extraction to convert audio to MFCC features")
        return np.array([]), np.array([])

    labels = [p.relative_to(root).parts[0] for p in paths]
    # Filter out any label not in CLASSES
    labels = [l for l in labels if l in CLASSES]

    print(f"Found {len(paths)} feature files in {root.name}")
    label_counts = {cls: sum(1 for l in labels if l == cls) for cls in CLASSES}
    for cls, count in label_counts.items():
        print(f"  {cls}: {count} files")

    return np.array(paths, dtype=object), np.array(labels, dtype=object)


def load_item(p: Path):
    """Load a single MFCC feature file"""
    x = np.load(p).astype(np.float32)  # (T,13) raw MFCC (no CMVN)
    # Robustly pad or trim to 66 frames
    target_frames = 66
    n_frames = x.shape[0]
    if n_frames < target_frames:
        pad_width = target_frames - n_frames
        x = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')
    elif n_frames > target_frames:
        x = x[:target_frames, :]
    # Defensive: if still wrong shape, force it
    x = x[:target_frames, :] if x.shape[0] > target_frames else x
    if x.shape[0] != target_frames:
        raise ValueError(f"MFCC shape error: got {x.shape}, expected ({target_frames}, 13) for file {p}")
    return x[:, None, :]  # (66,1,13)


def onehot(k):
    """Convert class index to one-hot encoding"""
    v = np.zeros((N_CLASSES,), np.float32)
    v[k] = 1.0
    return v


def balanced_paths(paths, labels):
    """Group paths by class for balanced sampling"""
    name_to_id = {n: i for i, n in enumerate(CLASSES)}
    y = np.array([name_to_id[s] for s in labels], np.int64)
    per = {c: [paths[i] for i in np.where(y == c)[0]] for c in range(N_CLASSES)}

    for c in per:
        if len(per[c]) == 0:
            print(f"WARNING: class {CLASSES[c]} is empty!")
        else:
            print(f"Class {CLASSES[c]}: {len(per[c])} samples for training")

    return per


def aug_time_jitter(x, p=0.25, max_shift=2):
    """Apply time jittering augmentation"""
    if np.random.rand() < p:
        dt = np.random.randint(-max_shift, max_shift + 1)
        x = np.roll(x, dt, axis=0)
    return x


def aug_noise(x, sigma=0.0025):
    """Apply noise augmentation"""
    return x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)


def make_train_ds(paths, labels, batch=192, shuffle=True):
    """Create balanced training dataset with augmentation"""
    per = balanced_paths(paths, labels)

    # Check if all classes have data
    empty_classes = [CLASSES[c] for c in range(N_CLASSES) if len(per[c]) == 0]
    if empty_classes:
        raise ValueError(f"Cannot train: empty classes {empty_classes}")

    def gen(c):
        plist = [str(p) for p in per[c]]
        if shuffle:
            random.shuffle(plist)
        for sp in plist:
            x = load_item(Path(sp))
            x = aug_time_jitter(x)
            x = aug_noise(x, 0.0025)
            yield x, onehot(c)

    dslist = [
        tf.data.Dataset.from_generator(
            lambda c=c: gen(c),
            output_signature=(
                tf.TensorSpec(INPUT_SHAPE, tf.float32),
                tf.TensorSpec((N_CLASSES,), tf.float32),
            ),
        ).repeat()
        for c in range(N_CLASSES)
    ]
    ds = tf.data.Dataset.sample_from_datasets(
        dslist, weights=[1 / N_CLASSES] * N_CLASSES
    )
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_eval_ds(paths, labels, batch=512):
    """Create evaluation dataset"""
    if len(paths) == 0:
        return None, np.array([])

    name_to_id = {n: i for i, n in enumerate(CLASSES)}
    y = np.array([name_to_id[s] for s in labels], np.int64)

    def gen():
        for sp, yy in zip(paths, y):
            yield load_item(Path(sp)), yy

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(INPUT_SHAPE, tf.float32),
            tf.TensorSpec((), tf.int64),
        ),
    )
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE), y


# ---------- Model (UltraTiny-DSCNN; esp-nn friendly) ----------
def ds_block(x, dw_k=5, stride=1, pw_ch=24, drop=0.0):
    """Depthwise separable convolution block"""
    # IMPORTANT: equal strides for Metal backend (stride, stride)
    x = layers.DepthwiseConv2D(
        (dw_k, 1), strides=(stride, stride), padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(pw_ch, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x


def build_model():
    """Build UltraTiny-DSCNN model for 4-class classification"""
    inp = keras.Input(shape=INPUT_SHAPE, name="input")  # (66,1,13)
    x = layers.Normalization(axis=-1, mean=CMVN_MEAN, variance=CMVN_VAR, name="cmvn")(inp)

    # Initial conv
    x = layers.Conv2D(12, (3, 1), padding="same", use_bias=False, name="conv_init")(x)
    x = layers.BatchNormalization(name="bn_init")(x)
    x = layers.ReLU(max_value=6.0, name="relu_init")(x)

    # Depthwise separable blocks
    x = ds_block(x, dw_k=5, stride=2, pw_ch=16, drop=0.05)  # 66->33 (time), freq stays 1
    x = ds_block(x, dw_k=5, stride=2, pw_ch=24, drop=0.05)  # 33->17
    x = ds_block(x, dw_k=5, stride=2, pw_ch=32, drop=0.05)  # 17->9

    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.10, name="dropout_final")(x)
    out = layers.Dense(N_CLASSES, name="classifier")(x)  # logits for 4 classes

    model = keras.Model(inp, out, name="UltraTinyDSCNN_KWS_4Class")
    return model


# ---------- Data Loading ----------
print("Loading dataset splits...")

train_paths, train_lbls = scan_split(FEAT_ROOT / "train")
val_paths, val_lbls = scan_split(FEAT_ROOT / "val")
test_paths, test_lbls = scan_split(FEAT_ROOT / "test")

if len(train_paths) == 0:
    print("ERROR: No training data found!")
    print("Make sure to run feature extraction on your Processed_Audio first")
    exit(1)

BATCH = 192
print(f"\nCreating datasets with batch size {BATCH}...")

train_ds = make_train_ds(train_paths, train_lbls, BATCH, shuffle=True)
val_raw, _ = make_eval_ds(val_paths, val_lbls) if len(val_paths) > 0 else (None, np.array([]))
test_raw, _ = make_eval_ds(test_paths, test_lbls) if len(test_paths) > 0 else (None, np.array([]))

if val_raw is not None:
    val_ds = val_raw.map(lambda x, y: (x, tf.one_hot(y, N_CLASSES)))
else:
    val_ds = None
    print("WARNING: No validation data found")


# ---------- LR schedule ----------
@tf.keras.utils.register_keras_serializable(package="kws")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay"""

    def __init__(self, base_lr=1e-3, warmup_steps=500, decay_steps=20000, alpha=0.15):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.alpha = float(alpha)
        self._cos = tf.keras.optimizers.schedules.CosineDecay(
            self.base_lr, self.decay_steps, alpha=self.alpha
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * tf.minimum(1.0, step / float(self.warmup_steps))
        return tf.where(
            step < self.warmup_steps, warm, self._cos(step - self.warmup_steps)
        )


# Training configuration
steps_per_epoch = int(np.ceil(len(train_paths) / BATCH))
TOTAL_EPOCHS = 36
print(f"Training configuration:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total epochs: {TOTAL_EPOCHS}")

lr = 1e-3  # Use fixed learning rate for robust training
opt = tf.keras.optimizers.Adam(lr)
loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)

# ---------- Build and Train Model ----------
print("\nBuilding model...")
model = build_model()
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")]
)

# Callbacks (using Keras ML libraries)
early = keras.callbacks.EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best weights
checkpoint = keras.callbacks.ModelCheckpoint(
    OUT_DIR / "best_model.keras",
    monitor="val_acc",
    mode="max",
    save_best_only=True,
    verbose=1
)

callbacks = [early]
if val_ds is not None:
    callbacks.append(checkpoint)

print("\nModel Architecture:")
print(model.summary())

print(f"\nStarting training...")
history = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)


# ---------- Evaluation Functions ----------
def preds_from_ds(m, ds):
    """Get predictions from dataset"""
    if ds is None:
        return np.array([])
    y_pred = []
    for batch_x, _ in ds:
        logits = m.predict_on_batch(batch_x)
        y_pred.append(np.argmax(logits, axis=1))
    return np.concatenate(y_pred, axis=0)


def ytrue_from_ds(ds):
    """Get true labels from dataset"""
    if ds is None:
        return np.array([])
    yy = []
    for _, y in ds:
        yy.append(y.numpy())
    return np.concatenate(yy, axis=0)


def f1_scores(y_true, y_pred, n_classes):
    """Calculate F1 scores per class and overall metrics"""
    f1_per = np.zeros((n_classes,), np.float32)
    supports = np.zeros((n_classes,), np.int64)

    for k in range(n_classes):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        f1_per[k] = f1
        supports[k] = np.sum(y_true == k)

    macro = float(np.mean(f1_per))
    weighted = float(np.sum(f1_per * (supports / np.maximum(1, np.sum(supports)))))

    return f1_per, macro, weighted, supports


# ---------- Model Evaluation ----------
print("\n" + "=" * 60)
print("EVALUATING MODEL PERFORMANCE")
print("=" * 60)

# Validation evaluation
if val_raw is not None:
    val_y_true = ytrue_from_ds(val_raw)
    val_y_pred = preds_from_ds(model, val_raw)
    val_f1_per, val_macro, val_weighted, val_supports = f1_scores(val_y_true, val_y_pred, N_CLASSES)
    val_acc = float(np.mean(val_y_pred == val_y_true))

    print(f"VALIDATION RESULTS:")
    print(f"  Overall Accuracy: {val_acc:.4f}")
    print(f"  Macro F1:        {val_macro:.4f}")
    print(f"  Weighted F1:     {val_weighted:.4f}")
    print(f"  Per-class results:")
    for i, (name, f1, support) in enumerate(zip(CLASSES, val_f1_per, val_supports)):
        print(f"    {name:>15}: F1={f1:.4f} (n={support})")

# Test evaluation
if test_raw is not None:
    test_y_true = ytrue_from_ds(test_raw)
    test_y_pred = preds_from_ds(model, test_raw)
    test_f1_per, test_macro, test_weighted, test_supports = f1_scores(test_y_true, test_y_pred, N_CLASSES)
    test_acc = float(np.mean(test_y_pred == test_y_true))

    print(f"\nTEST RESULTS:")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Macro F1:        {test_macro:.4f}")
    print(f"  Weighted F1:     {test_weighted:.4f}")
    print(f"  Per-class results:")
    for i, (name, f1, support) in enumerate(zip(CLASSES, test_f1_per, test_supports)):
        print(f"    {name:>15}: F1={f1:.4f} (n={support})")

# ---------- Full INT8 PTQ export ----------
print(f"\n{'=' * 60}")
print("EXPORTING QUANTIZED MODEL")
print("=" * 60)


def representative_gen(max_samples=800):
    """Generate representative samples for quantization"""
    print(f"Generating {max_samples} representative samples for quantization...")
    n = 0
    for p in train_paths[:max_samples]:  # Limit to available samples
        x = load_item(Path(p))[None, ...]  # [1,66,1,13] float
        yield [x]
        n += 1
        if n >= max_samples:
            break
    print(f"Generated {n} representative samples")


# Export quantized model
tmp = tempfile.mkdtemp()
try:
    # Create serving signature
    @tf.function(input_signature=[tf.TensorSpec([None, *INPUT_SHAPE], tf.float32)])
    def serving(x):
        return {"logits": model(x, training=False)}


    # Save model
    tf.saved_model.save(model, tmp, signatures={"serving_default": serving})

    # Convert to TensorFlow Lite with INT8 quantization
    conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    print("Converting model to INT8 TensorFlow Lite...")
    tfl = conv.convert()

    tflite_path = OUT_DIR / "kws_ultratiny_int8_4class.tflite"
    tflite_path.write_bytes(tfl)
    print(f"✓ Quantized model saved: {tflite_path}")

finally:
    shutil.rmtree(tmp, ignore_errors=True)



# Save TFLite model and print summary for software use
tflite_path = OUT_DIR / "kws_ultratiny_int8_4class.tflite"
print(f"\n{'=' * 60}")
print("EXPORT COMPLETE")
print("=" * 60)
print(f"✓ TensorFlow Lite model: {tflite_path}")
print(f"✓ Best model weights:    {OUT_DIR / 'best_model.keras'}")
model_size = tflite_path.stat().st_size
print(f"\nModel details:")
print(f"  Classes: {N_CLASSES} ({', '.join(CLASSES)})")
print(f"  Model size: {model_size:,} bytes ({model_size / 1024:.1f} KB)")
print(f"  Input shape: {INPUT_SHAPE}")
print(f"  Quantization: INT8")