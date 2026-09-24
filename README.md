# DiagnoX
# TremorTrack

### Hybrid Quantum-Classical Platform for Early Disease-Risk Assessment

TremorTrack is a smartphone-based longitudinal tremor monitoring platform designed to transform repeated movement measurements into an interpretable early-risk assessment.

## Key Features

- Smartphone accelerometer-based movement monitoring
- Signal processing and noise reduction
- Tremor feature extraction
- Personal baseline and longitudinal tracking
- Classical Machine Learning using SVM
- Quantum Machine Learning using VQC
- Classical vs Quantum model comparison
- Risk assessment
- Explainable trend reports
- Follow-up support

## System Workflow

Smartphone Motion Data
        ↓
Signal Processing
        ↓
Feature Extraction
        ↓
Personal Baseline
        ↓
Classical ML + Quantum ML
        ↓
Model Comparison
        ↓
Risk Assessment
        ↓
Longitudinal Report

## Technology Stack

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- PennyLane
- Android Studio
- Java/Kotlin
- Firebase

## Machine Learning

The prototype compares:

1. Classical ML
   - Support Vector Machine (SVM)

2. Quantum ML
   - Variational Quantum Classifier (VQC)
   - Quantum simulator using PennyLane

## Features Extracted

The prototype extracts:

- Movement RMS
- Signal variability
- Peak movement
- Dominant frequency

## Important Note

This repository contains a research/prototype implementation.

TremorTrack is intended for monitoring and screening support and is NOT a standalone medical diagnostic system.

The included demonstration dataset is simulated and must be replaced with properly validated biomedical data for meaningful model evaluation.
