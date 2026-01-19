print('[DEBUG] Script started')
try:
    import os
    import numpy as np
    import librosa
    import sounddevice as sd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import soundfile as sf
except Exception as e:
    print('[ERROR] Exception during imports:', e)
    import traceback
    traceback.print_exc()
    exit(1)

import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import soundfile as sf

# --- CONFIG ---

SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_LEN = SAMPLE_RATE  # 1 second
DATA_ROOT = os.path.join(os.getcwd(), 'Split_Audio')
CATEGORIES = ['on', 'off', 'silence']
SILENCE_DIR = os.path.join(DATA_ROOT, 'train', 'silence')

# --- DATA LOADING ---
def load_wavs_and_labels():
    X, y = [], []
    for idx, label in enumerate(CATEGORIES):
        folder = os.path.join(DATA_ROOT, 'train', label)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                path = os.path.join(folder, fname)
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)
                if len(audio) < AUDIO_LEN:
                    audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
                else:
                    audio = audio[:AUDIO_LEN]
                X.append(audio)
                y.append(idx)
    return np.array(X), np.array(y)

# --- FEATURE EXTRACTION ---
def extract_mfcc_batch(X):
    mfccs = [librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T for x in X]
    # Pad/truncate to 66 frames
    mfccs = [np.pad(m, ((0, max(0, 66 - m.shape[0])), (0, 0)), mode='constant')[:66] for m in mfccs]
    return np.stack(mfccs)

# --- MODEL ---
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(66, N_MFCC)),
        tf.keras.layers.Reshape((66, N_MFCC, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# --- SILENCE RECORDING ---
def record_silence_samples(num_samples=20, duration=1.0):
    os.makedirs(SILENCE_DIR, exist_ok=True)
    print(f"[LOG] Recording {num_samples} silence samples. Please stay quiet...")
    for i in range(num_samples):
        print(f"[LOG] Recording silence sample {i+1}/{num_samples}...")
        try:
            print("[LOG] Starting sd.rec...")
            recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            print("[LOG] Finished sd.rec.")
            audio = recording.flatten()
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            if len(audio) < AUDIO_LEN:
                audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
            else:
                audio = audio[:AUDIO_LEN]
            out_path = os.path.join(SILENCE_DIR, f'silence_{i+1:03d}.wav')
            print(f"[LOG] Saving to {out_path}...")
            sf.write(out_path, audio, SAMPLE_RATE)
            print(f"[LOG] Saved {out_path}.")
        except Exception as e:
            print(f"[ERROR] Exception during silence sample {i+1}: {e}")
    print("[LOG] Silence samples recorded.")

if not os.path.exists(SILENCE_DIR) or len(os.listdir(SILENCE_DIR)) < 5:
    print('[DEBUG] Checking silence samples...')
    print('[DEBUG] Not enough silence samples, recording...')
    record_silence_samples(num_samples=20, duration=1.0)
else:
    print('[DEBUG] Enough silence samples found, skipping recording.')

print('Loading data...')
X, y = load_wavs_and_labels()
print(f'Total samples: {len(X)}')
X_mfcc = extract_mfcc_batch(X)
X_train, X_val, y_train, y_val = train_test_split(X_mfcc, y, test_size=0.2, random_state=42, stratify=y)

print('Building model...')
model = build_model()
print('Training...')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# --- LIVE TESTING ---
def live_test():
    print('[DEBUG] Entered live_test()')
    print('Speak "on", "off", or stay silent (Ctrl+C to exit)...')
    # Test microphone access before loop
    try:
        print('[DEBUG] Testing microphone access...')
        test_recording = sd.rec(int(SAMPLE_RATE * 0.5), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print('[DEBUG] Microphone test recording shape:', test_recording.shape)
    except Exception as e:
        print('[ERROR] Microphone test failed:', e)
        return
    duration = 1.0
    while True:
        print('[DEBUG] Starting new live_test loop iteration')
        recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = recording.flatten()
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
        else:
            audio = audio[:AUDIO_LEN]
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
        if mfcc.shape[0] < 66:
            mfcc = np.pad(mfcc, ((0, 66 - mfcc.shape[0]), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:66]
        mfcc = np.expand_dims(mfcc, axis=0)
        print('[DEBUG] MFCC shape for prediction:', mfcc.shape)
        pred = np.argmax(model.predict(mfcc), axis=1)[0]
        print(f'Predicted: {CATEGORIES[pred]}')


if __name__ == '__main__':
    import traceback
    try:
        live_test()
        print('[LOG] live_test() finished (should not happen unless you exit).')
    except Exception as e:
        print('\nERROR: An exception occurred!')
        print(e)
        traceback.print_exc()
        input('Press Enter to exit...')
