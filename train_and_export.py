import numpy as np
import tensorflow as tf
from speech_commands_dataset_builder import Builder, WORDS, SAMPLE_RATE
from mfcc_features import extract_mfcc
from ds_cnn_model import build_ds_cnn
import tensorflow_datasets as tfds

print(tf.config.list_physical_devices('GPU'))

# Parameters
N_MFCC = 40  # More MFCCs for richer features
FRAME_LENGTH = 0.025
FRAME_STRIDE = 0.010
BATCH_SIZE = 64
EPOCHS = 2


# Restrict to only 'on', 'off', 'background_noise', 'unknown'
TARGET_KEYWORDS = ['on', 'off']
SPECIAL_LABELS = ['_silence_', '_unknown_']
SELECTED_LABELS = TARGET_KEYWORDS + SPECIAL_LABELS

builder = Builder()
builder.download_and_prepare()
dataset = builder.as_dataset(split='train', as_supervised=True)
val_dataset = builder.as_dataset(split='validation', as_supervised=True)
test_dataset = builder.as_dataset(split='test', as_supervised=True)


# Map integer label index to class name using builder.info.features['label'].names
def filter_selected(audio, label):
    # label is a numpy.int64 index
    label_idx = int(label)
    class_names = builder.info.features['label'].names
    return class_names[label_idx] in SELECTED_LABELS

def tf_filter_selected(audio, label):
    return tf.py_function(func=filter_selected, inp=[audio, label], Tout=tf.bool)

dataset = dataset.filter(tf_filter_selected)
val_dataset = val_dataset.filter(tf_filter_selected)
test_dataset = test_dataset.filter(tf_filter_selected)

# --- BEGIN: Balance classes in training set ---
import random
import numpy as np

def materialize_dataset(ds):
    audios = []
    labels = []
    class_names = builder.info.features['label'].names
    label_to_index = {name: idx for idx, name in enumerate(SELECTED_LABELS)}
    for audio, label in ds:
        label_idx = int(label.numpy())
        class_name = class_names[label_idx]
        # Only keep samples in SELECTED_LABELS, map others to _unknown_
        if class_name in label_to_index:
            mapped_label = label_to_index[class_name]
        else:
            mapped_label = label_to_index['_unknown_']
        audios.append(audio.numpy())
        labels.append(mapped_label)
    # Remove any samples with labels outside 0-3
    audios_filtered = []
    labels_filtered = []
    for audio, label in zip(audios, labels):
        if label in [0, 1, 2, 3]:
            audios_filtered.append(audio)
            labels_filtered.append(label)
    return np.array(audios_filtered, dtype=object), np.array(labels_filtered)

train_audios, train_labels = materialize_dataset(dataset)
class_names = builder.info.features['label'].names
label_to_index = {name: idx for idx, name in enumerate(SELECTED_LABELS)}
from collections import Counter
# Print class distribution before balancing
print(f"[DEBUG] Filtered label distribution: {Counter(train_labels)}")

# Balance all classes to the same count for robust learning
max_per_class = min(
    [np.sum(train_labels == i) for i in range(len(SELECTED_LABELS)) if np.sum(train_labels == i) > 0]
)
print(f"[DEBUG] Using {max_per_class} samples per class for balanced training.")
indices = []
for i in range(len(SELECTED_LABELS)):
    idx = np.where(train_labels == i)[0]
    if len(idx) > max_per_class:
        idx = np.random.choice(idx, max_per_class, replace=False)
    else:
        idx = np.random.choice(idx, max_per_class, replace=True)
    indices.extend(idx)
random.shuffle(indices)
balanced_audios = train_audios[indices]
balanced_labels = np.array([train_labels[i] for i in indices])
# Recreate tf.data.Dataset using from_generator for variable-length audio
print(f"[DEBUG] Number of balanced samples: {len(balanced_audios)}")
# Print class distribution in balanced_labels
from collections import Counter
print(f"[DEBUG] Balanced label distribution: {Counter(balanced_labels)}")
def gen_balanced():
    for i, (audio, label) in enumerate(zip(balanced_audios, balanced_labels)):
        if i < 5:
            print(f"[DEBUG] Yielding sample {i}, audio shape: {audio.shape}, label: {label}")
        yield audio, label
balanced_train_ds = tf.data.Dataset.from_generator(
    gen_balanced,
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
)
# --- END: Balance classes in training set ---

# Data augmentation functions
def add_noise(audio, noise_factor=0.005):
    audio = tf.cast(audio, tf.float32) / 32768.0  # scale int16 to [-1, 1]
    noise = tf.random.normal(shape=tf.shape(audio), mean=0.0, stddev=1.0, dtype=tf.float32)
    augmented = audio + noise_factor * noise
    return tf.clip_by_value(augmented, -1.0, 1.0)

def random_gain(audio, min_gain=0.8, max_gain=1.2):
    gain = tf.random.uniform([], min_gain, max_gain)
    return audio * gain

def time_stretch(audio, rate_range=(0.8, 1.2)):
    rate = tf.random.uniform([], rate_range[0], rate_range[1])
    audio_len = tf.shape(audio)[0]
    stretched = tf.signal.resample(audio, tf.cast(tf.cast(audio_len, tf.float32) / rate, tf.int32))
    stretched = tf.image.resize_with_crop_or_pad(tf.expand_dims(stretched, 1), audio_len, 1)
    return tf.squeeze(stretched, 1)

def time_shift(audio, shift_max=1600):
    audio = tf.cast(audio, tf.float32) / 32768.0  # ensure float32 for consistency
    shift = tf.random.uniform([], -shift_max, shift_max, dtype=tf.int32)
    return tf.roll(audio, shift, axis=0)

def augment(audio, label):
    # Randomly apply augmentation
    audio = tf.cast(audio, tf.float32) / 32768.0  # always work in float32 [-1, 1]
    audio = tf.cond(tf.random.uniform([]) > 0.5, lambda: add_noise(audio), lambda: audio)
    audio = tf.cond(tf.random.uniform([]) > 0.5, lambda: time_shift(audio), lambda: audio)
    audio = tf.cond(tf.random.uniform([]) > 0.5, lambda: random_gain(audio), lambda: audio)
        # Time stretching is disabled for TensorFlow 2.10.1 compatibility (no tf.signal.resample)
        # If you upgrade to TF 2.11+, you can re-enable this augmentation.
    return audio, label

# Preprocessing function for TFDS
def preprocess(audio, label):
    mfcc = extract_mfcc(audio, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, frame_length=FRAME_LENGTH, frame_stride=FRAME_STRIDE)
    # Normalize MFCCs (zero mean, unit variance)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    print(f"[DEBUG] TRAIN MFCC shape: {mfcc.shape}")
    print(f"[DEBUG] TRAIN MFCC sample values: {mfcc.flatten()[:10]}")
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
    return mfcc, label


# Remap label index to 0–3 for SELECTED_LABELS
def remap_label(label):
    # label is int, original index in builder.info.features['label'].names
    class_names = builder.info.features['label'].names
    label_name = class_names[int(label)]
    label_to_index = {name: idx for idx, name in enumerate(SELECTED_LABELS)}
    if label_name not in label_to_index:
        print(f"[DEBUG] remap_label: {label_name} not in SELECTED_LABELS, mapping to _unknown_")
        return np.int64(label_to_index['_unknown_'])
    print(f"[DEBUG] remap_label: {label_name} -> {label_to_index[label_name]}")
    return np.int64(label_to_index[label_name])

def tf_preprocess(audio, label):
    mfcc, label = tf.numpy_function(preprocess, [audio, label], [tf.float32, tf.int64])
    # Remap label to 0–3
    label = tf.numpy_function(remap_label, [label], tf.int64)
    mfcc.set_shape([None, N_MFCC, 1])
    label.set_shape([])
    return mfcc, label

# Specify padded shapes for variable-length MFCCs
padded_shapes = ([None, N_MFCC, 1], [])  # (MFCC shape, label shape)
train_ds = balanced_train_ds.map(tf_preprocess).shuffle(512).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes).prefetch(tf.data.AUTOTUNE)

# Debug: Take and print the first batch to ensure data is flowing
for batch in train_ds.take(1):
    mfccs, labels = batch
    print(f"[DEBUG] First batch MFCCs shape: {mfccs.shape}, labels: {labels.numpy()}")
    # Print a few (label, MFCC) pairs
    for i in range(min(5, mfccs.shape[0])):
        print(f"[DEBUG] Sample {i} label: {labels.numpy()[i]}, MFCC mean: {np.mean(mfccs[i].numpy())}, std: {np.std(mfccs[i].numpy())}")
    break
val_ds = val_dataset.map(tf_preprocess).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes).prefetch(tf.data.AUTOTUNE)
test_ds = test_dataset.map(tf_preprocess).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes).prefetch(tf.data.AUTOTUNE)


# Print class distribution of test set after preprocessing
print("Test set class distribution after preprocessing:")
labels = []
for x, y in test_ds:
    labels.append(np.argmax(y.numpy()))
from collections import Counter
print(Counter(labels))


# Print class distribution of train set after preprocessing
print("Train set class distribution after preprocessing:")
train_labels = []
for x, y in train_ds:
    train_labels.append(np.argmax(y.numpy()))
print(Counter(train_labels))

# Print a sample of one-hot encoded labels and true indices from train set
print("Sample one-hot encoded labels and true indices from train set:")
for i, (x, y) in enumerate(train_ds):
    print("One-hot:", y.numpy()[0], "Index:", np.argmax(y.numpy()[0]))
    if i >= 4:
        break

# Print class distribution of test set after preprocessing
print("Test set class distribution after preprocessing:")
test_labels = []
for x, y in test_ds:
    test_labels.append(np.argmax(y.numpy()))
print(Counter(test_labels))

# Print a sample of one-hot encoded labels and true indices from test set
print("Sample one-hot encoded labels and true indices from test set:")
for i, (x, y) in enumerate(test_ds):
    print("One-hot:", y.numpy()[0], "Index:", np.argmax(y.numpy()[0]))
    if i >= 4:
        break


from collections import Counter




# Check class balance in training set and compute class weights
from collections import Counter
labels = []
label_names = SELECTED_LABELS
label_to_index = {name: idx for idx, name in enumerate(label_names)}
class_names = builder.info.features['label'].names
for _, label in dataset:
    label_idx = int(label.numpy())
    class_name = class_names[label_idx]
    if class_name in label_to_index:
        labels.append(label_to_index[class_name])
print("[DEBUG] Label mapping:")
for idx, name in enumerate(label_names):
    print(f"  {idx}: {name}")
class_counts = Counter(labels)
print("[DEBUG] Training set class distribution:", class_counts)
num_classes = len(label_names)
total = sum(class_counts.values())
class_weights = {i: total/(num_classes*class_counts.get(i,1)) for i in range(num_classes)}
print("[DEBUG] Class weights:", class_weights)



# Model
input_shape = (None, N_MFCC, 1)


model = build_ds_cnn(input_shape, num_classes)

# Print model output logits for a batch of training data before training


# Print class distribution and sample labels for train_ds after preprocessing
print("[DEBUG] Train set class distribution after preprocessing and batching:")
train_label_counts = {}
for _, y_batch in train_ds:
    for label in y_batch.numpy():
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
print(train_label_counts)
print("[DEBUG] Sample train labels:", list(train_label_counts.keys())[:10])

# Print class distribution and sample labels for test_ds after preprocessing
print("[DEBUG] Test set class distribution after preprocessing and batching:")
test_label_counts = {}
for _, y_batch in test_ds:
    for label in y_batch.numpy():
        test_label_counts[label] = test_label_counts.get(label, 0) + 1
print(test_label_counts)
print("[DEBUG] Sample test labels:", list(test_label_counts.keys())[:10])

# Print model output logits and true class indices for a batch of training data
print("Model output logits for a batch of training data:")
for x_batch, y_batch in train_ds.take(1):
    logits = model(x_batch, training=False)
    print(logits.numpy()[:5])
    print("Predicted class indices:", np.argmax(logits.numpy(), axis=1)[:5])
    print("y_batch shape:", y_batch.shape)
    print("True class indices:", y_batch.numpy()[:5])
    break


# Check model loss function for label format consistency


# Callbacks for early stopping and best model saving
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]

# Train with class weights
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)


# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# Print confusion matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = []
y_pred = []
for batch in test_ds:
    mfccs, labels = batch
    preds = model.predict(mfccs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)
try:
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(SELECTED_LABELS))
    plt.xticks(tick_marks, SELECTED_LABELS, rotation=45)
    plt.yticks(tick_marks, SELECTED_LABELS)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"[DEBUG] Could not plot confusion matrix: {e}")


