# Outage Prediction

## Overview

This project is part of MSA Project Week at Georgia Institute of Technology that develops machine learning models to predict network outages using customer interaction data from a telecommunications company. Two main analysis notebooks are included, each exploring different time periods and modeling approaches.

---

## Notebooks

### 1. Outage_Prediction_7Days_Data.ipynb

**Dataset:** `0901to0907_agg0.csv` (September 1-7 data)
- **Rows:** 7,021,811
- **Columns:** 46 (pre-aggregated features)

#### Features
- **User Type Counts:** `ccmuser_count`, `cbuser_count`, `coxconnectuser_count`, `panoramicuser_count`, etc.
- **30-Minute Rolling Sums:** `connect_count_30min_sum`, `work_order_count_30min_sum`, etc.
- **Time Features:** `hour`, `day`, `day_of_week`
- **Target Variable:** `is_outage` (binary)

#### Model: Logistic Regression with 5-Fold Cross-Validation

| Metric | Class 0 (No Outage) | Class 1 (Outage) |
|--------|---------------------|------------------|
| Precision | 0.98 | 0.95 |
| Recall | 0.99 | 0.65 |
| F1-Score | 0.99 | 0.77 |

- **Mean AUC:** 0.944
- **Accuracy:** 97.7%

#### Confusion Matrix
| | Predicted No Outage | Predicted Outage |
|---|---------------------|------------------|
| **Actual No Outage** | 5,271,224 (TN) | 11,900 (FP) |
| **Actual Outage** | 117,055 (FN) | 217,269 (TP) |

#### Feature Importance (Top 5)
| Feature | Coefficient |
|---------|-------------|
| work_order_count | 10.33 |
| work_order_count_30min_sum | 9.82 |
| outage_pct | 4.72 |
| connect_count | 1.12 |
| cbuser_count | 0.73 |

---

### 2. Outage_Prediction_90Days_Data.ipynb

**Dataset:** `90d_sampled_agg.csv` (90-day sampled data)
- **Rows:** 9,999,715
- **Columns:** 44

#### Features
- **User Type Counts:** Various user interaction counts
- **30-Minute Rolling Sums:** Aggregated user activity features
- **Time Features:** Weekday indicators (0-6), time period (midnight-noon/noon-midnight UTC)
- **Line of Business:** `lob_C`, `lob_R` (Commercial/Residential)
- **Target Variable:** `outage` (binary)

#### Model 1: Logistic Regression with 5-Fold Cross-Validation

| Metric | Class 0 (No Outage) | Class 1 (Outage) |
|--------|---------------------|------------------|
| Precision | 0.96 | 0.94 |
| Recall | 1.00 | 0.46 |
| F1-Score | 0.98 | 0.62 |

- **Mean AUC:** 0.792
- **Accuracy:** 96%

#### Confusion Matrix (Logistic Regression)
| | Predicted No Outage | Predicted Outage |
|---|---------------------|------------------|
| **Actual No Outage** | 9,314,269 (TN) | 20,721 (FP) |
| **Actual Outage** | 360,014 (FN) | 304,711 (TP) |

#### PCA Analysis
- With PCA (95% variance): Average Accuracy = 0.944
- Without PCA: Average Accuracy = 0.962

#### Model 2: Neural Network (Keras)

**Architecture:**
- Input Layer: 41 features
- Hidden Layer 1: 64 neurons, ReLU activation
- Hidden Layer 2: 32 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation

**Training:** 10 epochs, batch size 256, validation split

#### Model Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.9620 | 0.9370 | 0.4585 | 0.7281 |
| Neural Network | 0.9629 | 0.9461 | 0.4680 | 0.8236 |

---

## Key Findings

1. **Work Order Count is the Strongest Predictor:** Across both datasets, `work_order_count` and related features show the highest predictive power for outages.

2. **Class Imbalance Challenge:** Both datasets show significant class imbalance (outages are rare events), leading to high precision but relatively lower recall for the outage class.

3. **7-Day Model Outperforms:** The 7-day model achieves better overall AUC (0.944) compared to the 90-day model (0.792), potentially due to more focused temporal patterns.

4. **Neural Networks Provide Marginal Improvement:** For the 90-day data, the neural network slightly outperforms logistic regression in AUC (0.8236 vs 0.7281) with similar accuracy.

5. **PCA Trade-off:** Using PCA for dimensionality reduction reduces accuracy from 96.2% to 94.4%, suggesting the original features contain valuable information.

---

## Requirements

```python
# Standard library
import os
import psutil
from datetime import datetime

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

# Deep Learning
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
```

---
