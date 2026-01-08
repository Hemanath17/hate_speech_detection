# Multilingual Hate Speech Detection System

A comprehensive, production-ready hate speech detection system that identifies and categorizes harmful content across five languages: English, Tamil, Hindi, Chinese (Mandarin), and Spanish. This system leverages state-of-the-art transformer models to achieve high accuracy in detecting hate speech across diverse linguistic and cultural contexts.

## Table of Contents

- [About the Project](#about-the-project)
- [Why Hate Speech Detection is Needed](#why-hate-speech-detection-is-needed)
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Technology Stack](#technology-stack)
- [NLP Algorithms and Models](#nlp-algorithms-and-models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Browser Extension](#browser-extension)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)

## About the Project

This project addresses the critical need for automated hate speech detection in multilingual online environments. With the exponential growth of user-generated content across social media platforms, forums, and comment sections, manual moderation has become impractical. This system provides real-time, accurate detection of hate speech across multiple languages, enabling platforms to maintain safer online communities.

The system is designed to handle the complexities of multilingual content, including native script processing, cultural context understanding, and language-specific hate speech patterns. It achieves high accuracy through a combination of advanced deep learning models, comprehensive datasets, and optimized detection thresholds.

## Why Hate Speech Detection is Needed

Hate speech poses significant threats to online communities and society at large:

1. **Psychological Impact**: Hate speech can cause severe emotional distress, anxiety, and trauma to targeted individuals and communities.

2. **Social Division**: It perpetuates stereotypes, fuels discrimination, and creates divisions within communities.

3. **Platform Safety**: Online platforms struggle to moderate billions of posts daily, requiring automated solutions to maintain safe environments.

4. **Legal Compliance**: Many jurisdictions require platforms to actively monitor and remove hate speech content.

5. **User Experience**: Hate speech degrades the quality of online discourse and drives away users seeking constructive interactions.

6. **Multilingual Challenge**: The global nature of the internet requires detection systems that work across multiple languages and cultural contexts.

This project provides a scalable, accurate solution to these challenges, supporting five major languages and achieving production-grade performance.

## Project Overview

The Multilingual Hate Speech Detection System is a comprehensive solution consisting of:

- **Backend API Server**: Flask-based REST API serving a fine-tuned mBERT model for hate speech detection
- **Browser Extension**: Real-time detection and highlighting of hate speech in web browsers
- **Training Pipeline**: Complete infrastructure for model training, evaluation, and optimization
- **Data Processing**: Tools for dataset creation, augmentation, and preprocessing
- **Model Optimization**: Threshold tuning and performance optimization for each supported language

The system processes text input, detects the language automatically, applies language-specific detection models, and returns confidence scores along with classification results.

## How It Works

### Detection Pipeline

1. **Text Input**: User provides text through API endpoint or browser extension
2. **Language Detection**: System automatically identifies the language (English, Tamil, Hindi, Chinese, Spanish)
3. **Preprocessing**: Text is cleaned, normalized, and tokenized according to language-specific rules
4. **Model Inference**: Fine-tuned mBERT model processes the text and generates probability scores
5. **Threshold Application**: Language-specific optimized thresholds are applied to determine hate speech classification
6. **Keyword Override**: Strong hate keywords trigger additional validation for borderline cases
7. **Result Generation**: System returns classification (hate/neutral), confidence score, and language information

### Browser Extension Workflow

1. **Content Monitoring**: Extension monitors all input fields and comment sections on web pages
2. **Real-time Detection**: As users type, text is analyzed after a debounce period (500ms)
3. **Visual Highlighting**: Detected hate speech is highlighted in red with visual indicators
4. **Pre-submission Warning**: Before posting, users are warned if hate speech is detected
5. **Comment Scanning**: Existing comments on pages are scanned and highlighted automatically

### Training Process

1. **Data Collection**: Aggregates datasets from multiple sources (ETHOS, Davidson, HateXplain, OLID)
2. **Data Augmentation**: Generates synthetic samples using back-translation and paraphrasing
3. **Preprocessing**: Native script processing, text cleaning, and language-specific pattern matching
4. **Model Fine-tuning**: Fine-tunes mBERT on multilingual dataset with language-specific optimization
5. **Threshold Tuning**: Optimizes decision thresholds per language to maximize F1 scores
6. **Evaluation**: Comprehensive testing across all languages with detailed metrics

## Features

- **Multilingual Support**: Detects hate speech in English, Tamil, Hindi, Chinese (Mandarin), and Spanish
- **High Accuracy**: Achieves 95.67% accuracy and 94.91% F1-macro score on test set
- **Language-Specific Optimization**: Custom thresholds and preprocessing for each language
- **Real-time Detection**: Browser extension provides instant feedback as users type
- **Visual Highlighting**: Red highlighting and badges for easy identification of hate speech
- **Pre-submission Blocking**: Warns users before posting hate speech content
- **Comment Section Scanning**: Automatically scans and highlights hate speech in existing comments
- **RESTful API**: Easy integration with existing applications
- **Production Ready**: Optimized for deployment with error handling and logging

## Technical Architecture

### System Components

1. **Backend API Server** (`src/api/app.py`)
   - Flask-based REST API
   - Model loading and inference
   - CORS-enabled for browser extension
   - Health check endpoints
   - Batch processing support

2. **Browser Extension** (`browser_extension/`)
   - Content scripts for real-time monitoring
   - Visual highlighting system
   - Form interception for pre-submission warnings
   - Comment section scanning
   - Settings and configuration UI

3. **Model Training Pipeline** (`src/models/`)
   - Training scripts for different architectures
   - Evaluation and metrics calculation
   - Threshold optimization
   - Model checkpointing

4. **Data Processing** (`src/data/`)
   - Dataset collection and aggregation
   - Data augmentation (back-translation, paraphrasing)
   - Preprocessing and cleaning
   - Language-specific pattern matching

### Data Flow

```
User Input → Language Detection → Preprocessing → mBERT Model → 
Threshold Application → Keyword Validation → Result (Hate/Neutral + Confidence)
```

## Technology Stack

### Core Technologies

- **Python 3.12**: Primary programming language
- **PyTorch 2.0+**: Deep learning framework for model training and inference
- **Transformers (Hugging Face)**: Pre-trained model library and tokenization
- **Flask**: Web framework for API server
- **Flask-CORS**: Cross-origin resource sharing for browser extension

### Machine Learning Libraries

- **scikit-learn**: Data preprocessing, train/test splitting, metrics calculation
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **tqdm**: Progress bars for training and evaluation

### NLP Libraries

- **Transformers**: BERT, DistilBERT, mBERT model implementations
- **Tokenizers**: Fast tokenization for multiple languages
- **NLTK**: Natural language processing utilities (optional)

### Development Tools

- **Git**: Version control
- **Virtual Environment**: Python environment isolation
- **Jupyter Notebooks**: Data analysis and experimentation (optional)

### Browser Extension

- **JavaScript (ES6+)**: Content scripts and extension logic
- **Chrome Extensions API**: Browser integration
- **Manifest V3**: Modern extension architecture

## NLP Algorithms and Models

### Primary Model: mBERT (Multilingual BERT)

**Model**: `bert-base-multilingual-cased`

**Architecture**:
- Transformer-based encoder with 12 layers
- 768-dimensional hidden states
- 110M parameters
- Supports 104 languages including all 5 target languages

**Why mBERT**:
- Pre-trained on multilingual corpora, providing cross-lingual understanding
- Handles code-switching and mixed-language content
- Strong performance on low-resource languages
- Unified architecture for all languages reduces deployment complexity

**Fine-tuning Process**:
- Binary classification head (hate/neutral)
- Language-specific token prefixes for better multilingual understanding
- Custom learning rate scheduling with warmup
- Gradient clipping for training stability
- Early stopping based on validation F1 score

### Alternative Models (Used in Development)

**DistilBERT** (`distilbert-base-uncased`):
- Lightweight BERT variant (66M parameters)
- Faster inference, lower memory footprint
- Used for initial experiments and binary classification

**mBART** (`facebook/mbart-large-50-many-to-many-mmt`):
- Multilingual translation model
- Used for data augmentation via back-translation
- Supports 50 languages for translation tasks

### NLP Techniques Applied

1. **Tokenization**:
   - WordPiece tokenization for English
   - Character-level tokenization for Chinese
   - Subword tokenization for Tamil, Hindi, Spanish
   - Language-specific tokenizers for optimal performance

2. **Text Preprocessing**:
   - Unicode normalization for native scripts
   - Special character handling for each language
   - URL and email removal
   - Whitespace normalization

3. **Language Detection**:
   - Pattern-based detection using Unicode ranges
   - Character frequency analysis
   - Script-based identification (Devanagari, Tamil, Chinese characters)

4. **Feature Engineering**:
   - Language-specific hate keyword patterns
   - Context-aware detection
   - Confidence score calibration

5. **Threshold Optimization**:
   - Per-language threshold tuning using grid search
   - ROC curve analysis
   - F1 score maximization
   - Precision-recall trade-off optimization

### Training Algorithms

- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 2e-5 with linear warmup scheduling
- **Loss Function**: Cross-entropy loss for binary classification
- **Regularization**: Dropout (0.1), gradient clipping (1.0)
- **Early Stopping**: Based on validation F1 score with patience of 3 epochs

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Git
- 8GB+ RAM (16GB recommended for training)
- 10GB+ free disk space (for models and datasets)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/hatespeech-detection.git
cd hatespeech-detection
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv hate_env
source hate_env/bin/activate  # On Windows: hate_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install API server dependencies
pip install -r src/api/requirements.txt
```

### Step 4: Download Pre-trained Model

The fine-tuned mBERT model is included in the repository. If you need to retrain:

```bash
# Model will be automatically downloaded on first use
# Or manually download from Hugging Face:
# transformers-cli download bert-base-multilingual-cased
```

### Step 5: Verify Installation

```bash
# Test API server
python src/api/app.py

# In another terminal, test the API
curl http://localhost:5000/health
```

Expected output:
```json
{"status":"healthy","model_loaded":true,"device":"cpu"}
```

## Usage

### Starting the API Server

```bash
# Activate virtual environment
source hate_env/bin/activate

# Start the server
python src/api/app.py

# Or use the convenience script
./src/api/start_server.sh
```

The API will be available at `http://localhost:5000`

### Using the API

#### Single Text Detection

```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "You are stupid and worthless"}'
```

Response:
```json
{
  "is_hate": true,
  "confidence": 0.833,
  "hate_probability": 0.833,
  "neutral_probability": 0.167,
  "language": "eng",
  "threshold_used": 0.4,
  "keyword_override": true
}
```

#### Batch Detection

```bash
curl -X POST http://localhost:5000/api/detect-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "I hate you"]}'
```

### Using the Browser Extension

1. **Load Extension in Chrome**:
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `browser_extension` folder

2. **Load Extension in Firefox**:
   - Open `about:debugging`
   - Click "This Firefox"
   - Click "Load Temporary Add-on"
   - Select `browser_extension/manifest.json`

3. **Start API Server** (required for extension to work):
   ```bash
   python src/api/app.py
   ```

4. **Use the Extension**:
   - Navigate to any website
   - Type in input fields - hate speech will be highlighted in real-time
   - Existing comments will be scanned and highlighted
   - Pre-submission warnings will appear when posting hate speech

### Python API Usage

```python
import requests

def detect_hate_speech(text):
    response = requests.post(
        'http://localhost:5000/api/detect',
        json={'text': text}
    )
    return response.json()

# Example usage
result = detect_hate_speech("You are stupid")
print(f"Is Hate: {result['is_hate']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Language: {result['language']}")
```

## Project Structure

```
Hatespeech/
├── browser_extension/          # Browser extension files
│   ├── manifest.json          # Extension manifest
│   ├── background/            # Service worker
│   ├── content/               # Content scripts
│   ├── popup/                 # Extension popup UI
│   └── assets/                # Icons and resources
├── configs/                   # Configuration files
│   └── config.py             # Model and data paths
├── Data/                      # Datasets
│   ├── collected_datasets/    # Collected public datasets
│   ├── Ethos-Hate-Speech-Dataset/  # ETHOS dataset
│   └── *.csv                 # Processed datasets
├── models/                    # Trained models
│   ├── mbert_improved_chinese_multilingual/  # Production model
│   └── optimized_thresholds.json  # Language-specific thresholds
├── notebooks/                 # Jupyter notebooks for analysis
├── src/
│   ├── api/                   # API server
│   │   ├── app.py            # Flask application
│   │   ├── requirements.txt   # API dependencies
│   │   └── start_server.sh   # Startup script
│   ├── data/                 # Data processing
│   │   ├── preprocessing.py  # Text preprocessing
│   │   ├── data_collector.py # Dataset collection
│   │   ├── data_augmenter.py # Data augmentation
│   │   └── ...
│   ├── models/               # Model training scripts
│   │   ├── distilbert_model.py  # DistilBERT implementation
│   │   ├── training_*.py     # Training scripts
│   │   ├── retrain_improved_chinese_mbert.py  # Retraining script
│   │   └── tune_mbert_thresholds.py  # Threshold optimization
│   └── utils/                # Utility functions
├── requirements.txt          # Python dependencies
├── SETUP_GUIDE.md           # Detailed setup instructions
└── README.md               # This file
```

## Performance Metrics

### Overall Performance

- **Test Accuracy**: 95.67%
- **Test F1-Macro**: 94.91%
- **Test Precision**: 94.90%
- **Test Recall**: 94.92%

### Language-Specific Performance

| Language | F1 Score | Accuracy | Samples |
|----------|----------|----------|---------|
| English  | 96.87%   | 94.93%   | 5,184   |
| Tamil    | 100.00%  | 100.00%  | 131     |
| Hindi    | 97.56%   | 99.59%   | 244     |
| Chinese  | 96.96%   | 97.92%   | 626     |
| Spanish  | 96.00%   | 98.94%   | 284     |

### Model Specifications

- **Model Size**: 692 MB
- **Inference Speed**: ~50-100ms per text (CPU)
- **Supported Languages**: 5 (English, Tamil, Hindi, Chinese, Spanish)
- **Max Input Length**: 128 tokens
- **Batch Processing**: Supported via API

## Browser Extension

The browser extension provides real-time hate speech detection directly in web browsers. It works on all major websites including YouTube, Twitter/X, Reddit, Facebook, and any site with comment sections or input fields.

### Extension Features

- Real-time detection as users type
- Visual highlighting (red for hate speech, yellow for borderline)
- Pre-submission warnings before posting
- Automatic comment section scanning
- Language auto-detection
- Configurable sensitivity settings
- Detection history and statistics

### Extension Architecture

- **Content Scripts**: Monitor and modify webpage content
- **Background Service Worker**: Handles extension lifecycle
- **Popup UI**: Settings and statistics interface
- **API Integration**: Communicates with backend server

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns server status and model information.

#### Single Detection
```
POST /api/detect
Content-Type: application/json

Body: {"text": "your text here"}
```

#### Batch Detection
```
POST /api/detect-batch
Content-Type: application/json

Body: {"texts": ["text1", "text2", ...]}
```

### Response Format

```json
{
  "is_hate": true,
  "confidence": 0.833,
  "hate_probability": 0.833,
  "neutral_probability": 0.167,
  "language": "eng",
  "threshold_used": 0.4,
  "keyword_override": false
}
```

### Error Handling

- **400 Bad Request**: Invalid input (missing text, empty text)
- **500 Internal Server Error**: Model loading failure or prediction error

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Follow installation instructions
2. Install development dependencies:
   ```bash
   pip install pytest black flake8
   ```
3. Run tests (when available)
4. Follow code style guidelines

### Areas for Contribution

- Additional language support
- Model performance improvements
- Browser extension enhancements
- Documentation improvements
- Bug fixes and optimizations

## Dataset Information

The system uses multiple datasets:

- **ETHOS Dataset**: Multi-label hate speech dataset with 8 categories
- **Davidson Dataset**: Twitter-based hate speech dataset (24,783 samples)
- **HateXplain Dataset**: Hate speech with explanations
- **OLID Dataset**: Offensive Language Identification Dataset
- **Tatoeba**: Multilingual sentence pairs for augmentation

Total training data: 32,342 samples across 5 languages.

## Model Training

To retrain the model with your own data:

```bash
# Prepare your dataset in CSV format with 'text' and 'isHate' columns
# Then run:
python src/models/retrain_improved_chinese_mbert.py
```

Training parameters can be adjusted in the script or via configuration files.

## Troubleshooting

### API Server Issues

- **Port already in use**: Change port in `src/api/app.py` or kill existing process
- **Model loading fails**: Check model path and file permissions
- **CORS errors**: Verify Flask-CORS is installed and configured

### Browser Extension Issues

- **Extension not working**: Check if API server is running on localhost:5000
- **No highlights appearing**: Verify extension is enabled in browser settings
- **API connection errors**: Check browser console (F12) for detailed errors

### Performance Issues

- **Slow inference**: Consider using GPU for faster processing
- **High memory usage**: Reduce batch size or use smaller model variant
- **Extension lag**: Increase debounce delay in content script



