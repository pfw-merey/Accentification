# Accentification: English Accent Identification Using Machine Learning

## Overview
This project identifies American and British English accents using machine learning. It extracts MFCC and NCC features from audio files and trains a TensorFlow model to classify accents.

## Directory Structure
- `data/`: Contains audio files organized by accent labels.
- `models/`: Stores trained models.
- `notebooks/`: Includes exploratory data analysis notebooks.
- `src/`: Python source files for data loading, feature extraction, model training, and evaluation.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. [Optional] Train a model

   Place your dataset in the `data/` directory with subfolders `american/` and `british/`, then run:
   
   ```bash
   python src/train.py
   ```
3. [Optional] Evaluate the model
   ```bash
   python src/evaluate.py
   ```
4. Run the model
   ```bash
   python predict.py --model models/v1_balanced.keras --file /path/to/your/file.mp3
   ```
