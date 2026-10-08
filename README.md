# SHL Grammar Scoring 2026 — Audio Regression

## Overview

This repository contains my solution for the SHL Hiring Assessment 2026 grammar scoring challenge.

The task is to predict a continuous grammar score from 0 to 5 for spoken-audio recordings.

## Dataset

- Training recordings: 769
- Test recordings: 216
- Audio format: mono WAV, 16 kHz
- Target: continuous grammar score from 0 to 5

## Approach

The solution uses an audio-based machine learning pipeline:

1. Validate train/test metadata and audio-file matching.
2. Extract acoustic speech features with Librosa.
3. Add fast speech-activity and energy-dynamics features.
4. Compare several regression models using 5-fold cross-validation.
5. Select an Extra Trees Regressor using the full 51-feature representation.
6. Generate out-of-fold predictions and apply a simple linear calibration.
7. Retrain the final model on all training samples.
8. Generate predictions for the 216 test recordings.

## Features

The final feature set contains 51 acoustic and speech-activity features, including:

- MFCC statistics
- Chroma statistics
- Mel-spectrogram statistics
- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Zero-crossing statistics
- RMS energy statistics
- Duration
- Speech-activity and silence features
- Energy-dynamics features

## Model

The final model is an Extra Trees Regressor with:

- 500 trees
- `min_samples_leaf = 2`
- `max_features = 0.8`
- `random_state = 42`

## Validation Results

Five-fold validation of the tested models:

| Model | Mean RMSE | Mean Pearson |
|---|---:|---:|
| Random Forest | 0.7893 | 0.7637 |
| HistGradientBoosting | 0.8041 | 0.7511 |
| Extra Trees (41 features) | 0.7637 | 0.7821 |
| Extra Trees (51 features) | **0.7618** | **0.7835** |

Out-of-fold evaluation of the final configuration:

- Baseline RMSE: 0.7634
- Pearson correlation: 0.7894
- Calibrated RMSE: 0.7601
- Calibrated Pearson correlation: 0.7894

Training RMSE on all 769 training samples:

**0.1045**

## Kaggle Submission

The final prediction file contains:

- 216 rows
- Columns: `filename`, `label`
- 216/216 test filenames matched
- No missing predictions
- Predictions constrained to the 0–5 target range

Public Kaggle score for Version 1:

**0.7249**

## Repository Contents

`shl-grammar-scoring-2026.ipynb` — complete notebook containing data validation, feature extraction, model experiments, validation, visualizations, and final submission generation.

## Reproducibility

The notebook is intended to be executed in the Kaggle competition environment where the challenge dataset is available. The dataset and audio files are not included in this repository.
