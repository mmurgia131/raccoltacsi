# CSI-Based People Counting System

## Overview

This repository contains a complete pipeline for **WiFi CSI (Channel State Information) based people counting** using machine learning. The system can detect the number of people (0-6) in an environment using WiFi channel characteristics from a MIMO setup.

## Features

- **Strict MIMO Synchronization**: Ensures all 4 receiver links are temporally aligned
- **Class Aggregation**: Merges classes by person count (not by activity modality)
- **Intelligent Balancing**: SMOTE for rare classes, class weights for natural distribution
- **5-Fold Cross-Validation**: Rigorous evaluation replacing single train/test split
- **Counting-Specific Metrics**: MAE, accuracy within ±1/±2 persons
- **Comprehensive Analysis**: Per-class metrics, systematic error detection

## System Architecture

### Dataset Classes

The system aggregates original classes by person count:

- **0** → 0 people (absence)
- **1D, 1S** → 1 person (dynamic/static merged)
- **2D** → 2 people
- **3D** → 3 people
- **4D** → 4 people
- **5D** → 5 people
- **6D** → 6 people
- **8D** → Excluded (insufficient samples: ~262)

### MIMO Configuration

The system requires synchronized data from 4 receiver links:
- `rx1_mac1`: Receiver 1, Transmitter 1
- `rx2_mac1`: Receiver 2, Transmitter 1
- `rx1_mac2`: Receiver 1, Transmitter 2
- `rx2_mac2`: Receiver 2, Transmitter 2

## Installation

```bash
# Clone the repository
git clone https://github.com/mmurgia131/raccoltacsi.git
cd raccoltacsi

# Install dependencies
pip install -r requirements.txt
```

## Pipeline Modules

### 1. `preprocessing_csi_counting.py`

**Purpose**: Load, parse, and synchronize CSI data from 4 MIMO links.

**Features**:
- Parses JSON CSI strings to complex arrays
- Extracts amplitude and phase features
- Applies Hampel filter for outlier removal
- Strict timestamp synchronization across all 4 links
- Discards non-synchronized samples

**Usage**:
```bash
python preprocessing_csi_counting.py \
    --rx1_mac1 data/rx1_mac1.csv \
    --rx2_mac1 data/rx2_mac1.csv \
    --rx1_mac2 data/rx1_mac2.csv \
    --rx2_mac2 data/rx2_mac2.csv \
    --output preprocessed_data.csv \
    --time_window 50
```

**Parameters**:
- `--time_window`: Max time difference (ms) for synchronization (default: 50)
- `--no_hampel`: Disable Hampel filter
- `--savgol`: Enable Savitzky-Golay smoothing

### 2. `dataset_balancing.py`

**Purpose**: Aggregate classes by person count and apply SMOTE to rare classes.

**Features**:
- Aggregates 1D/1S → 1 person, 2D → 2 people, etc.
- Excludes class 8D (insufficient samples)
- Applies SMOTE to classes with <50 samples
- Generates synthetic samples (target: 50 per class)
- Calculates class weights for model training
- Produces distribution visualizations and reports

**Usage**:
```bash
python dataset_balancing.py \
    --input preprocessed_data.csv \
    --output balanced_data.csv \
    --label_column label \
    --smote_threshold 50 \
    --k_neighbors 3 \
    --viz_output distribution.png \
    --report_output balancing_report.txt
```

**Outputs**:
- Balanced dataset CSV
- Class distribution visualization (PNG)
- Balancing methodology report (TXT)

### 3. `modeling_counting.py`

**Purpose**: Train models using 5-Fold Stratified Cross-Validation.

**Features**:
- 5-Fold Stratified CV (replaces 80/20 split)
- Multiple models: XGBoost, LightGBM, RandomForest
- Class weight support for imbalanced data
- Per-class metrics: precision, recall, F1-score
- Counting metrics: MAE, accuracy ±1, ±2 persons
- Confusion matrices for each fold

**Usage**:
```bash
python modeling_counting.py \
    --input balanced_data.csv \
    --label_column label \
    --model xgboost \
    --n_folds 5 \
    --output_dir results/
```

**Model Options**:
- `xgboost`: XGBoost Classifier (recommended)
- `lightgbm`: LightGBM Classifier
- `randomforest`: Random Forest Classifier

**Outputs**:
- CV results report (TXT)
- Confusion matrices visualization (PNG)

### 4. `evaluation_counting.py`

**Purpose**: Systematic error analysis per class.

**Features**:
- Per-class MAE calculation
- Error distribution analysis
- Systematic pattern detection (over/underestimation)
- Adjacent class confusion analysis
- Comprehensive visualizations

**Usage**:
```bash
python evaluation_counting.py \
    --predictions predictions.csv \
    --true_column y_true \
    --pred_column y_pred \
    --output_prefix evaluation
```

**Outputs**:
- Error distribution plots (PNG)
- Detailed evaluation report (TXT)

## Complete Workflow Example

```bash
# Step 1: Preprocess and synchronize MIMO data
python preprocessing_csi_counting.py \
    --rx1_mac1 raw_data/rx1_mac1.csv \
    --rx2_mac1 raw_data/rx2_mac1.csv \
    --rx1_mac2 raw_data/rx1_mac2.csv \
    --rx2_mac2 raw_data/rx2_mac2.csv \
    --output preprocessed.csv

# Step 2: Balance dataset with class aggregation and SMOTE
python dataset_balancing.py \
    --input preprocessed.csv \
    --output balanced.csv \
    --viz_output distribution.png

# Step 3: Train model with 5-Fold CV
python modeling_counting.py \
    --input balanced.csv \
    --model xgboost \
    --n_folds 5 \
    --output_dir results/

# Step 4: Evaluate systematic errors
python evaluation_counting.py \
    --predictions results/predictions.csv \
    --output_prefix results/evaluation
```

## Methodological Choices

### 1. Class Aggregation
**Rationale**: Merging by person count (not activity) creates a more robust counting system. The distinction between "1D" (1 person dynamic) and "1S" (1 person static) is less relevant for count estimation.

### 2. Excluding 8D
**Rationale**: With only ~262 samples (0.1%), class 8D is statistically unreliable and would introduce noise. The system focuses on 0-6 persons, which covers most practical scenarios.

### 3. SMOTE for Rare Classes
**Rationale**: Classes with <50 samples benefit from synthetic oversampling. Using k_neighbors=3 is appropriate for small original sample sizes.

### 4. Class Weights
**Rationale**: Maintaining natural distribution with class weights allows the model to learn the true data distribution while still handling imbalance.

### 5. 5-Fold Stratified CV
**Rationale**: Provides more reliable estimates than a single 80/20 split, especially important given class imbalance. Stratification ensures each fold maintains class proportions.

### 6. Strict MIMO Synchronization
**Rationale**: Ensures all 4 links contribute data from the same time instant, critical for spatial diversity in MIMO systems.

## Metrics

### Counting-Specific Metrics

1. **MAE (Mean Absolute Error)**: Average counting error in number of people
2. **Exact Accuracy**: Percentage of perfect count predictions
3. **Accuracy ±1 person**: Tolerance for ±1 person error
4. **Accuracy ±2 persons**: Tolerance for ±2 person error

### Per-Class Metrics

- **Precision**: How many predicted counts were correct for each class
- **Recall**: How many actual counts were detected for each class
- **F1-Score**: Harmonic mean of precision and recall

## Expected Performance

Based on the methodology:
- **MAE**: Expected 0.3-0.8 persons
- **Exact Accuracy**: 60-80%
- **Accuracy ±1**: 85-95%
- **Accuracy ±2**: 95-99%

Performance varies by class due to original data imbalance.

## Data Format

### Input CSV Format (Raw CSI)

Required columns:
```
time, type, id, mac, rssi, rate, sig_mode, mcs, bandwidth,
smoothing, not_sounding, aggregation, stbc, fec_coding, sgi,
noise_floor, ampdu_cnt, channel, secondary_channel, 
local_timestamp, ant, sig_len, rx_state, len, first_word, data
```

The `data` column contains JSON-encoded CSI values (I/Q pairs).

### Label Format

Labels should be strings: `'0'`, `'1D'`, `'1S'`, `'2D'`, `'3D'`, `'4D'`, `'5D'`, `'6D'`, `'8D'`

## Troubleshooting

### Issue: "No synchronized samples found"
**Solution**: Increase `--time_window` parameter or check timestamp alignment in raw data.

### Issue: "Not enough samples for SMOTE"
**Solution**: Reduce `--k_neighbors` or `--smote_threshold` parameters.

### Issue: "Class 8D warnings"
**Solution**: This is expected - class 8D is intentionally excluded.

## Contributing

Contributions are welcome! Please ensure:
1. Code follows existing style
2. Documentation is updated
3. Tests pass (if applicable)

## License

Apache License 2.0

## Citation

If you use this code in your research, please cite:

```bibtex
@software{raccoltacsi,
  author = {Murgia, M.},
  title = {CSI-Based People Counting System},
  year = {2026},
  url = {https://github.com/mmurgia131/raccoltacsi}
}
```

## Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This system is designed for research and educational purposes. Performance may vary based on environment, hardware setup, and data quality.
