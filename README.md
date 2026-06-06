# 🫁 AI-Powered Chest X-Ray Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/)

> Leveraging Convolutional Neural Networks (CNNs) to classify chest X-ray images into four categories: **COVID-19**, **Normal**, **Pneumonia**, and **Tuberculosis**. This project achieves **95.14% accuracy**, supporting early diagnosis and clinical decision-making in respiratory disease detection.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Environment Variables](#-environment-variables)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Custom CNN Architecture**: Deep learning model specifically designed for chest X-ray classification

- **High Accuracy**: Achieves **95.14% accuracy** on test data with robust performance metrics

- **Comprehensive Evaluation**: 
  - Accuracy, Precision, Recall, F1-Score metrics
  - Training/validation loss and accuracy curves
  - Model performance tracking across epochs

- **Multi-Class Classification**: 
  - COVID-19 detection
  - Pneumonia identification
  - Tuberculosis recognition
  - Normal chest X-ray classification

- **Data Preprocessing**: Robust pipeline with normalization and proper train/validation/test splitting

- **Visualization Tools**: 
  - Sample image visualization from training/validation/testing sets
  - Training history plots
  - Model performance comparison charts

---

## 🎯 Model Performance

| Model | Test Accuracy | Precision | Recall | F1-Score | Loss |
|-------|--------------|-----------|---------|----------|------|
| **Custom CNN** | **95.14%** | **95.36%** | **94.81%** | **95.11%** | **0.1632** |

### Training Performance
- **Best Validation Accuracy**: 95.62% (Epoch 10)
- **Best Validation Loss**: 0.1635 (Epoch 6)
- **Total Training Time**: ~11 epochs with early stopping
- **Final Training Accuracy**: 99.35%

**Best Model**: Custom CNN with **95.14% test accuracy** 🏆

---

## 📊 Dataset

### Source
The project uses the CXR Dataset from Kaggle:

- **CXR Data Set** by Reflex7
  - [Dataset Link](https://www.kaggle.com/datasets/reflex7/cxr-data-set)

### Dataset Structure
```
/kaggle/input/cxr-data-set/
├── Covid/
├── Normal/
├── Pneumonia/
└── Tuberculosis/

/kaggle/working/split_data/
├── train/
│   ├── Covid/
│   ├── Normal/
│   ├── Pneumonia/
│   └── Tuberculosis/
├── validation/
│   ├── Covid/
│   ├── Normal/
│   ├── Pneumonia/
│   └── Tuberculosis/
└── test/
    ├── Covid/
    ├── Normal/
    ├── Pneumonia/
    └── Tuberculosis/
```

### Classes
- **COVID-19**: Chest X-rays showing COVID-19 infection patterns
- **Normal**: Healthy chest X-ray scans without abnormalities
- **Pneumonia**: X-rays displaying pneumonia-related lung infections
- **Tuberculosis**: Chest radiographs showing tuberculosis manifestations

### Dataset Split
- **Training Set**: 57,832 images (80%)
- **Validation Set**: 7,232 images (10%)
- **Test Set**: 7,232 images (10%)

### Class Distribution
| Class | Label | Total Images |
|-------|-------|--------------|
| COVID-19 | 0 | 18,074 |
| Normal | 1 | 18,074 |
| Pneumonia | 2 | 18,074 |
| Tuberculosis | 3 | 18,074 |

### Preprocessing
- Images resized to **224x224** pixels
- Normalization: pixel values scaled to [0, 1]
- Stratified train/validation/test split (80/10/10)
- Batch size: **32**

---

## 📁 Project Structure

```
CXR_IMAGE_CLASSIFICATION/
│
├── result/
│   └── CNN_Result/
│       ├── CNN.png                    # Model architecture visualization
│       ├── model_accuracy.png         # Training/validation accuracy plot
│       ├── result.png                 # Final results comparison
│       ├── val_accuracy.png           # Validation accuracy over epochs
│       └── val_loss.png              # Validation loss over epochs
│
├── Data_Visualization/
│   ├── testing.png                    # Sample test images
│   ├── train.png                      # Sample training images
│   └── validation.png                 # Sample validation images
│
├── src/
│   └── model/
│       └── model.txt.txt              # Model architecture details
│
├── cxr-image-classification-cnn-95-acc.ipynb   # Main training notebook
├── models.txt                         # Links to trained models
├── dataset.txt                        # Dataset download information
├── kaggle_notebook.txt                # Link to Kaggle notebook
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── LICENSE                            # Project license
└── README.md                          # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- 16GB+ RAM (recommended)
- 10GB+ free disk space

### Installation Steps

#### 1. Install Python using MiniConda

Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)

Create a new environment:
```bash
conda create -n CXR_Classification python=3.8
```

Activate the environment:
```bash
conda activate CXR_Classification
```

#### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/cxr-image-classification.git
cd cxr-image-classification
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download Dataset

**Option A: Using Kaggle API**
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('reflex7/cxr-data-set')"
```

**Option B: Manual Download**
- Download dataset from the link in `dataset.txt`
- Extract to `/kaggle/input/cxr-data-set/` directory

#### 5. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` file:
```env
PROJECT_VERSION=1.0
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.001
```

---

## 💻 Usage

### Training the Model

Run the main training notebook:
```bash
jupyter notebook cxr-image-classification-cnn-95-acc.ipynb
```

Or run as a Python script:
```bash
python cxr-image-classification-cnn-95-acc.py
```

### Model Inference

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('CNN_best_model.keras')

# Load and preprocess image
img_path = 'path/to/chest_xray.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
classes = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']
result = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2f}%")
```

### Evaluating the Model

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocess function
def cnn_preprocess(x):
    return x/255.0

# Create test generator
test_datagen = ImageDataGenerator(preprocessing_function=cnn_preprocess)
test_generator = test_datagen.flow_from_directory(
    '/kaggle/working/split_data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate model
results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall: {results[3]*100:.2f}%")
print(f"Test F1-Score: {results[4]*100:.2f}%")
```

---

## 🔧 Technical Details

### Architecture Overview

#### Custom CNN
```
Input (224x224x3)
→ Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2)
→ Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2)
→ Conv2D(128, 3×3, ReLU) + MaxPooling2D(2×2)
→ Flatten
→ Dropout(0.4)
→ Dense(128, ReLU)
→ Dense(64, ReLU)
→ Dense(64, ReLU)
→ Dense(4, Softmax)
```

**Total Parameters**: ~3.5M parameters

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 (with early stopping) |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Metrics | Accuracy, Precision, Recall, F1-Score |

### Callbacks
- **EarlyStopping**: 
  - Monitor: `val_loss`
  - Patience: 5 epochs
  - Restore best weights: True

- **ModelCheckpoint**: 
  - Filepath: `CNN_best_model.keras`
  - Monitor: `val_loss`
  - Save best only: True

- **ReduceLROnPlateau**: 
  - Monitor: `val_loss`
  - Factor: 0.5
  - Patience: 3 epochs
  - Min LR: 1e-7

### Hardware Requirements
- **GPU**: NVIDIA Tesla P100-PCIE-16GB (used in training)
- **CUDA**: Compute Capability 6.0
- **cuDNN**: Version 9.3.0
- **RAM**: 16GB recommended
- **Storage**: 10GB for datasets and models
- **Framework**: TensorFlow with XLA optimization

---

## 📈 Results

### Training History

The model was trained for 11 epochs before early stopping:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1 | 77.76% | 89.85% | 0.5720 | 0.3056 | 0.001 |
| 2 | 91.43% | 93.57% | 0.2480 | 0.1961 | 0.001 |
| 5 | 96.64% | 94.80% | 0.0942 | 0.1931 | 0.001 |
| 6 | 97.39% | 94.99% | 0.0771 | **0.1635** | 0.001 |
| 10 | 99.09% | **95.62%** | 0.0261 | 0.2244 | 0.0005 |
| 11 | 99.35% | 95.34% | 0.0186 | 0.2381 | 0.0005 |

### Key Findings

1. **Fast Convergence**: Model achieved >90% accuracy within 2 epochs
2. **Excellent Generalization**: Test accuracy (95.14%) close to validation accuracy (95.62%)
3. **Balanced Performance**: High precision (95.36%) and recall (94.81%) indicate balanced predictions
4. **Low Overfitting**: Validation loss remained stable with proper regularization
5. **Robust F1-Score**: 95.11% F1-score demonstrates strong overall performance

### Training Curves

The training shows:
- Smooth accuracy improvement across epochs
- Controlled validation loss with minimal overfitting
- Effective learning rate reduction strategy
- Proper convergence without premature stopping

---

## 🌐 Environment Variables

Create a `.env` file with the following variables:

```env
# Project Configuration
PROJECT_NAME=CXR_Image_Classification
PROJECT_VERSION=1.0

# Model Parameters
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.001

# Paths
BASE_DIR=/kaggle/input/cxr-data-set
OUTPUT_DIR=/kaggle/working/split_data
TRAIN_PATH=/kaggle/working/split_data/train
VALID_PATH=/kaggle/working/split_data/validation
TEST_PATH=/kaggle/working/split_data/test
MODEL_SAVE_PATH=/kaggle/working/

# Data Split Ratios
TEST_SIZE=0.1
VAL_SIZE=0.1
TRAIN_SIZE=0.8

# Training Configuration
EARLY_STOPPING_PATIENCE=5
REDUCE_LR_PATIENCE=3
REDUCE_LR_FACTOR=0.5
MIN_LEARNING_RATE=1e-7

# Callbacks
MONITOR_METRIC=val_loss
SAVE_BEST_ONLY=True
RESTORE_BEST_WEIGHTS=True
```

---

## 📚 Resources

### Trained Models
Trained model (.keras format) available:
- **CNN_best_model.keras** - Best performing model from training

See `models.txt` for download links.

### Kaggle Notebook
Complete interactive notebook with all visualizations:
- See `kaggle_notebook.txt` for the link to the Kaggle notebook

### Dataset
Download link available in `dataset.txt`:
- CXR Data Set (Reflex7)

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras API Reference](https://keras.io/api/)
- [COVID-19 Chest X-ray Database](https://github.com/ieee8023/covid-chestxray-dataset)
- [WHO Tuberculosis Resources](https://www.who.int/health-topics/tuberculosis)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to functions and classes
- Update README.md for significant changes
- Test code before submitting PR
- Ensure all tests pass

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Abdelhady Ali** - *Initial work* - [MyGitHub](https://github.com/Abdelhady-22)

---

## 🙏 Acknowledgments

- **Dataset**: Thanks to Reflex7 for providing the comprehensive CXR dataset on Kaggle
- **Kaggle**: For providing the platform and computational resources (Tesla P100 GPU)
- **TensorFlow/Keras**: For the powerful deep learning framework
- **Medical Community**: For inspiring this work to assist in respiratory disease diagnosis
- **Open Source Community**: For various tools and libraries used in this project

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email**: (abdelhady2322005@gmail.com)
- **LinkedIn**: (https://www.linkedin.com/in/abdelhady-ali-940761316)
- **GitHub**: (https://github.com/Abdelhady-22)
- **Kaggle**: (https://www.kaggle.com/abdulhadialimohamed)

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The model should not be used as a substitute for professional medical diagnosis. Chest X-ray interpretation requires expertise from qualified radiologists and healthcare professionals. Always consult medical experts for accurate diagnosis and treatment decisions.

**Important Notes:**
- This tool is not FDA approved
- Not intended for clinical use without proper validation
- Should not replace professional medical judgment
- Results may vary depending on image quality and patient conditions

---



### Dataset Citation
```bibtex
@dataset{reflex7_cxr_2024,
  title={CXR Data Set},
  author={Reflex7},
  year={2024},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/reflex7/cxr-data-set}
}
```

---

## 🔮 Future Improvements

- [ ] Implement data augmentation for improved generalization
- [ ] Experiment with transfer learning (ResNet, EfficientNet, DenseNet)
- [ ] Add Grad-CAM visualization for model interpretability
- [ ] Multi-model ensemble for improved accuracy
- [ ] Deploy as web application using Streamlit or Flask
- [ ] Add API endpoints for production integration
- [ ] Expand dataset with more diverse X-ray images
- [ ] Implement severity classification for detected diseases
- [ ] Add patient demographic integration
- [ ] Create mobile application for point-of-care diagnosis

---

<div align="center">

**Made with ❤️ for advancing medical AI and improving healthcare accessibility**

⭐ Star this repo if you find it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Abdelhady-22/cxr-image-classification?style=social)](https://github.com/Abdelhady-22/CXR_image_classification)
[![GitHub forks](https://img.shields.io/github/forks/Abdelhady-22/cxr-image-classification?style=social)](https://github.com/Abdelhady-22/CXR_image_classification)
[![GitHub watchers](https://img.shields.io/github/watchers/Abdelhady-22/cxr-image-classification?style=social)](https://github.com/Abdelhady-22/CXR_image_classification)

</div>