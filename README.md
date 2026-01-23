# Depthwise Separable CNN Audio Processing

A compact and efficient keyword spotting (KWS) system that classifies audio into four categories: **ON**, **OFF**, **SILENCE**, and **UNKNOWN** words. Built with a depthwise separable CNN architecture, this project is optimized for resource-constrained environments while maintaining high accuracy.

## 🎯 Project Overview

This project implements a complete end-to-end pipeline for audio classification:

1. **Audio Preprocessing** - Split and process raw audio files into normalized 1-second segments
2. **Feature Extraction** - Convert audio to MFCC (Mel-Frequency Cepstral Coefficients) features
3. **Model Training** - Train a lightweight UltraTiny-DSCNN model with 4 classes
4. **Live Classification** - Real-time audio classification from microphone input
5. **Model Export** - Export to TensorFlow Lite INT8 quantized format for embedded deployment

## 📊 Dataset

The project uses the **Speech Commands Dataset v2** with 13 different word categories:

**Target Keywords (Primary Classes):**
- `on` - 3,076 training samples
- `off` - 2,996 training samples

**Background/Silence:**
- `_background_noise_` - 5,057 training samples (environmental noise)

**Unknown Words (Non-Target Speech):**
- `backward`, `cat`, `down`, `five`, `forward`, `four`, `learn`, `right`, `six`, `stop`, `up`, `visual`, `zero`
- 31,515 training samples total (consolidated into "unknown" category)

**Data Split:**
- Training: 42,644 samples
- Validation: 5,320 samples
- Test: 5,345 samples

## 🏗️ Model Architecture

**UltraTiny-DSCNN**
- Optimized for embedded systems with minimal parameters
- Depthwise separable convolutions for efficiency
- CMVN (Channel-wise Mean and Variance Normalization) preprocessing
- 4 output classes with softmax activation

**Model Details:**
- Input shape: `(66, 1, 13)` - 66 time frames × 13 MFCC coefficients
- Model size: 13.0 KB (TensorFlow Lite INT8)
- Quantization: Full INTEGER8 Post-Training Quantization (PTQ)

**Performance Metrics:**

*Validation Set:*
- Overall Accuracy: **91.8%**
- F1 Scores:
  - ON: 0.74
  - OFF: 0.69
  - SILENCE: 0.96
  - UNKNOWN: 0.96

*Test Set:*
- Overall Accuracy: **90.2%**
- F1 Scores:
  - ON: 0.73
  - OFF: 0.71
  - SILENCE: 0.86
  - UNKNOWN: 0.95

## 📂 Project Structure

```
.
├── Split_Audio/              # Raw audio files split by class
│   ├── train/
│   ├── val/
│   └── test/
├── KWS_MFCC32_UltraTiny/     # MFCC features extracted from audio
│   ├── train/
│   ├── val/
│   ├── test/
│   └── cmvn_mfcc_train.npz   # Normalization statistics
├── artifacts_kws_ultratiny_int8/  # Trained model outputs
│   ├── best_model.keras
│   └── kws_ultratiny_int8_4class.tflite
├── MFCCPrep.py               # Feature extraction script
├── ModelTraining.py          # Model training script
├── on_off_classifier.py      # Live classification script
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
python 3.9+
tensorflow 2.x
librosa
sounddevice
soundfile
numpy
scikit-learn
```

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

2. Install dependencies:
```bash
pip install tensorflow librosa sounddevice soundfile numpy scikit-learn
```

### Step 1: Extract MFCC Features

Process audio files and extract MFCC features:

```bash
python MFCCPrep.py
```

This will:
- Load audio files from `Split_Audio/`
- Extract MFCC features with parameters:
  - Sample rate: 16 kHz
  - MFCC coefficients: 13
  - FFT window: 512
  - Hop length: 160 samples
  - Mel bins: 32
- Save features to `KWS_MFCC32_UltraTiny/`
- Compute CMVN statistics from training data

### Step 2: Train the Model

Train the UltraTiny-DSCNN with 4 classes:

```bash
python ModelTraining.py
```

This will:
- Load MFCC features from `KWS_MFCC32_UltraTiny/`
- Train the model for up to 36 epochs with early stopping
- Evaluate on validation and test sets
- Export to `artifacts_kws_ultratiny_int8/`
  - `best_model.keras` - Keras model with best validation accuracy
  - `kws_ultratiny_int8_4class.tflite` - INT8 quantized TFLite model

### Step 3: Live Classification

Run real-time audio classification:

```bash
python on_off_classifier.py
```

The classifier will:
- Load the trained model
- Listen for 1-second audio samples
- Display detected classification with confidence scores
- Show all class probabilities in a visual format

## 🔍 Understanding the Output

When you run `on_off_classifier.py`, you'll see:

```
╔════════════════════════════════════════╗
║  🟢 Detected: ON         87.3%        ║
╚════════════════════════════════════════╝
  ┌────────────────────────────────────────┐
  │ 🟢 ON       [████████████████░░░░] 87.3% │
  │ 🔴 OFF      [█░░░░░░░░░░░░░░░░░░]  3.2% │
  │ ⚫ SILENCE   [█░░░░░░░░░░░░░░░░░░]  2.1% │
  │ 🟡 UNKNOWN   [█░░░░░░░░░░░░░░░░░░]  7.4% │
  └────────────────────────────────────────┘
```

- **Main Box**: Highest confidence classification
- **Confidence Bars**: Visual representation of all class probabilities
- **Icons**: Color-coded for quick visual recognition
  - 🟢 ON (Green)
  - 🔴 OFF (Red)
  - ⚫ SILENCE (Black)
  - 🟡 UNKNOWN (Yellow)

## 💡 Key Features

✅ **4-Class Classification**
- Accurately distinguishes between target keywords and unknown words
- Previously misclassified unknown words as on/off - now properly categorized

✅ **Lightweight & Portable**
- Model size: 13 KB (TensorFlow Lite format)
- Can run on embedded devices (microcontrollers, IoT devices)
- INT8 quantization for inference efficiency

✅ **Robust Feature Extraction**
- MFCC features matched between training and inference
- CMVN normalization for consistent results
- 1-second fixed-length audio processing

✅ **Professional Interface**
- Real-time audio classification with visual feedback
- Confidence scores for all predictions
- Clear "Speak now" prompts for user guidance

## 🔧 Advanced Usage

### Custom Audio Processing

To add your own audio files:

1. Place audio files in appropriate class folders under `Split_Audio/`
2. Run `MFCCPrep.py` to extract features
3. Run `ModelTraining.py` to retrain

### Model Export for Deployment

The TensorFlow Lite model is ready for embedded deployment:

```python
import tensorflow as tf

# Load and use the quantized model
interpreter = tf.lite.Interpreter('artifacts_kws_ultratiny_int8/kws_ultratiny_int8_4class.tflite')
interpreter.allocate_tensors()
```

### Hyperparameter Tuning

Edit these in `ModelTraining.py`:
- `N_CLASSES`: Number of output classes
- `BATCH`: Training batch size
- `TOTAL_EPOCHS`: Maximum training epochs
- `lr`: Learning rate

## 📈 Training Details

- **Optimizer**: Adam with base learning rate 1e-3
- **Loss Function**: Categorical crossentropy with 5% label smoothing
- **Metrics**: Categorical accuracy
- **Early Stopping**: Patience of 8 epochs on validation accuracy
- **Data Augmentation**: Time jittering and Gaussian noise injection
- **Balanced Sampling**: Equal weight to all classes during training

## 🎓 About Depthwise Separable Convolutions

Depthwise separable convolutions decompose standard convolutions into:

1. **Depthwise Convolution**: Spatial convolution on each input channel separately
2. **Pointwise Convolution**: 1×1 convolution to combine channel information

**Benefits:**
- Fewer parameters (typically 8-9× reduction)
- Faster computation
- Better for mobile/embedded deployment
- Maintains or improves accuracy compared to standard convolutions

## ⚙️ MFCC Feature Extraction

**MFCC (Mel-Frequency Cepstral Coefficients)** capture the characteristics of human hearing:

1. **Mel-Scale**: Frequencies are warped to match human perception
2. **Cepstral**: Decorrelated coefficients via DCT
3. **Coefficients**: 13 MFCC features per time frame

**Parameters used:**
- Window size (N_FFT): 512 samples (~32 ms at 16 kHz)
- Hop length: 160 samples (~10 ms)
- Mel bins: 32
- Frequency range: 20 Hz - 8 kHz

## 🐛 Troubleshooting

**Issue**: Model not found when running classifier
- **Solution**: Run `ModelTraining.py` first to generate the model

**Issue**: Poor classification accuracy
- **Solution**: Ensure audio files are in the correct class folders and have consistent sample rates (16 kHz)

**Issue**: Microphone not detected
- **Solution**: Check microphone permissions and ensure audio input device is connected

## 📝 License

This project uses the Google Speech Commands Dataset v2, which is licensed under CC-BY-4.0.

## 🔗 References

- [Google Speech Commands Dataset](https://github.com/google-research/google-research/tree/master/gsc_wgan)
- [Depthwise Separable Convolutions - MobileNets](https://arxiv.org/abs/1704.04861)
- [MFCC - Mel-Frequency Cepstral Coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

---

**Version**: 1.0  
**Last Updated**: January 23, 2026
