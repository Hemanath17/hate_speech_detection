"""
Configuration file for Hate Speech Detection System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"
BROWSER_EXT_DIR = PROJECT_ROOT / "browser_extension"

# Model configurations
MODEL_CONFIG = {
    "distilbert_model": "distilbert-base-uncased",
    "mbart_model": "facebook/mbart-large-50-many-to-many-mmt",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "patience": 3,
    "weight_decay": 0.01,
    "gradient_clip": 1.0
}

# Language configurations
LANGUAGES = {
    "en": {"code": "en_XX", "name": "English"},
    "ta": {"code": "ta_IN", "name": "Tamil"},
    "hi": {"code": "hi_IN", "name": "Hindi"},
    "es": {"code": "es_XX", "name": "Spanish"},
    "zh": {"code": "zh_CN", "name": "Mandarin"}
}

# Multi-label categories
CATEGORIES = [
    "violence",
    "directed_vs_generalized", 
    "gender",
    "race",
    "national_origin",
    "disability",
    "religion",
    "sexual_orientation"
]

# Performance targets
PERFORMANCE_TARGETS = {
    "overall_f1": 0.90,
    "min_category_f1": 0.80,
    "max_inference_time": 2.0,  # seconds
    "max_memory_usage": 2.0,    # GB
    "min_confidence": 0.7
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    "synonym_replacement": True,
    "back_translation": True,
    "random_insertion": True,
    "random_swap": True,
    "random_deletion": True,
    "augmentation_factor": 2.0  # 2x original data
}

# Browser extension settings
EXTENSION_CONFIG = {
    "supported_sites": [
        "twitter.com",
        "youtube.com", 
        "reddit.com",
        "instagram.com",
        "facebook.com",
        "tiktok.com",
        "linkedin.com"
    ],
    "detection_interval": 1000,  # milliseconds
    "min_text_length": 10,
    "max_text_length": 1000
}

# API settings
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "workers": 1,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "timeout": 30
}

# Data paths
DATA_PATHS = {
    'ethos_multilabel': 'Data/Ethos_Dataset_Multi_Label.csv',
    'ethos_binary': 'Data/Ethos-Hate-Speech-Dataset/ethos/ethos_data/Ethos_Dataset_Binary.csv',
    'ethos_multilabel_full': 'Data/Ethos-Hate-Speech-Dataset/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv',
    'tamil': 'Data/tam_sentences_detailed.tsv.bz2',
    'hindi': 'Data/hin_sentences_detailed.tsv.bz2',
    'spanish': 'Data/spa_sentences_detailed.tsv.bz2',
    'mandarin': 'Data/cmn_sentences_detailed.tsv.bz2'
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/hate_speech_detection.log"
}
