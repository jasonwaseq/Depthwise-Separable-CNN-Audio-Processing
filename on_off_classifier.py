print('Loading KWS Classifier...')
try:
    import os
    import numpy as np
    import librosa
    import sounddevice as sd
    import tensorflow as tf
except Exception as e:
    print(f'[ERROR] Failed to import dependencies: {e}')
    exit(1)

# --- CONFIG ---
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 32
FMIN = 20
FMAX = SAMPLE_RATE // 2
AUDIO_LEN = SAMPLE_RATE  # 1 second

# Categories must match the order used in ModelTraining.py
CATEGORIES = ['on', 'off', '_background_noise_', 'unknown']
DISPLAY_NAMES = {'on': 'ON', 'off': 'OFF', '_background_noise_': 'SILENCE', 'unknown': 'UNKNOWN'}
DISPLAY_COLORS = {'on': '🟢', 'off': '🔴', '_background_noise_': '⚫', 'unknown': '🟡'}

# Path to the trained model
MODEL_PATH = os.path.join(os.getcwd(), 'artifacts_kws_ultratiny_int8', 'best_model.keras')

# --- LOAD TRAINED MODEL ---
if not os.path.exists(MODEL_PATH):
    print(f'❌ Model not found at {MODEL_PATH}')
    print('   Please run ModelTraining.py first to train the model.')
    exit(1)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(MODEL_PATH)
print('✅ Model loaded successfully!\n')


def extract_mfcc_features(audio):
    """Extract MFCC features matching the training pipeline"""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        center=True,
        pad_mode='constant'
    )
    
    mfcc = mfcc.T  # (time_frames, n_mfcc)
    
    target_frames = 66
    if mfcc.shape[0] < target_frames:
        pad_width = target_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    elif mfcc.shape[0] > target_frames:
        mfcc = mfcc[:target_frames, :]
    
    mfcc = mfcc[:, np.newaxis, :]  # (66, 1, 13)
    return mfcc.astype(np.float32)


def print_header():
    print('=' * 50)
    print('        🎤 KEYWORD SPOTTING CLASSIFIER 🎤')
    print('=' * 50)
    print('  Say "ON", "OFF", or any other word')
    print('  Press Ctrl+C to exit')
    print('=' * 50)
    print()


def print_result(category, confidence):
    display_name = DISPLAY_NAMES.get(category, category)
    icon = DISPLAY_COLORS.get(category, '⚪')
    bar_length = int(confidence * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    print(f'\r  {icon} {display_name:8} [{bar}] {confidence:6.1%}', end='', flush=True)


def print_all_scores(confidence):
    print('\n  ┌────────────────────────────────────────┐')
    for i, cat in enumerate(CATEGORIES):
        name = DISPLAY_NAMES.get(cat, cat)
        icon = DISPLAY_COLORS.get(cat, '⚪')
        bar_len = int(confidence[i] * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f'  │ {icon} {name:8} [{bar}] {confidence[i]:5.1%} │')
    print('  └────────────────────────────────────────┘\n')


# --- LIVE TESTING ---
def live_test():
    print_header()
    
    # Test microphone access
    try:
        test_recording = sd.rec(int(SAMPLE_RATE * 0.2), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        print(f'❌ Microphone error: {e}')
        return
    
    duration = 1.0
    while True:
        print('🎙️  Speak now...', end='', flush=True)
        recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print('\r' + ' ' * 20 + '\r', end='', flush=True)  # Clear the "Speak now" line
        
        audio = recording.flatten()
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        if len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
        else:
            audio = audio[:AUDIO_LEN]
        
        mfcc = extract_mfcc_features(audio)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        logits = model.predict(mfcc, verbose=0)
        pred = np.argmax(logits, axis=1)[0]
        confidence = tf.nn.softmax(logits[0]).numpy()
        
        category = CATEGORIES[pred]
        display_name = DISPLAY_NAMES.get(category, category)
        icon = DISPLAY_COLORS.get(category, '⚪')
        
        # Show the main prediction prominently
        print(f'  ╔════════════════════════════════════════╗')
        print(f'  ║  {icon} Detected: {display_name:8}  {confidence[pred]:6.1%}        ║')
        print(f'  ╚════════════════════════════════════════╝')
        
        # Show all confidence scores
        print_all_scores(confidence)


if __name__ == '__main__':
    try:
        live_test()
    except KeyboardInterrupt:
        print('\n\n👋 Goodbye!')
    except Exception as e:
        print(f'\n❌ Error: {e}')
        input('Press Enter to exit...')
