# 🫁 LUNA16 Lung Nodule Classification

> **ROI-based machine learning pipeline for LUNA16 CT scans**  
> From nodule annotations and ROI extraction to feature selection, model comparison, cross-validation, and scan-level leakage-free evaluation.

---

## 🌟 Project Overview

This project explores a classical machine-learning approach for distinguishing **nodule vs. non-nodule ROI samples** extracted from the **LUNA16 lung CT dataset**.

The goal is **not lung cancer diagnosis**. Instead, the project focuses on classification of candidate image regions based on simple first-order intensity statistics.

The complete workflow was developed step by step:

**LUNA16 annotations → ROI extraction → feature extraction → feature selection → model comparison → cross-validation → leakage investigation → scan-level validation → final evaluation**

The main motivation was to build a pipeline that is not only capable of achieving good predictive performance, but also demonstrates why **data leakage prevention and correct validation strategy are essential in medical imaging**.

---

## 🎯 Main Objectives

- 🫁 Work with LUNA16 CT/nodule data
- 🎯 Identify positive nodule regions and corresponding ROI samples
- 🧩 Convert ROI images into numerical features
- 📊 Compare different classical machine-learning algorithms
- 🔎 Investigate which features are most informative
- 🔁 Evaluate models using cross-validation
- 🛡️ Detect and prevent scan-level data leakage
- 🧪 Perform feature selection inside training folds
- 📈 Compare models using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- 📚 Build a reproducible and explainable ML pipeline

---

## 🗂️ Dataset

### LUNA16

The project is based on the **LUNA16** dataset, which was created from the **LIDC-IDRI** collection.

LUNA16 contains:

- 🩻 **888 CT scans**
- 🫁 Annotated pulmonary nodules
- 📄 Nodule annotations and candidate locations
- 🗃️ 10 data subsets (`subset0`–`subset9`)
- 📏 CT scans with slice thickness greater than 2.5 mm excluded from the challenge dataset
- 👨‍⚕️ Reference nodules based on radiologist annotations

The LUNA16 reference standard considers nodules of at least 3 mm that were accepted by at least 3 of the 4 radiologists.

### Official resources

- [LUNA16 Data](https://luna16.grand-challenge.org/Data/)
- [LUNA16 Procedure](https://luna16.grand-challenge.org/Procedure/)
- [LUNA16 Description](https://luna16.grand-challenge.org/Description/)
- [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)

### 📜 Dataset Licensing

LUNA16 data are distributed under **CC BY 4.0**.  
The underlying LIDC-IDRI collection is distributed under **CC BY 3.0**.

Please consult the official dataset pages for the complete licensing and citation requirements.

---

# 🧠 Project Pipeline

```text
                    LUNA16 / LIDC-IDRI
                           │
                           ▼
                  Nodule / Candidate Data
                           │
                           ▼
                    ROI Extraction
                           │
                           ▼
                 Numerical Feature Table
                           │
                           ▼
                 Feature Selection
                           │
                           ▼
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Model Comparison          Cross-Validation
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Leakage Investigation
                           │
                           ▼
              Scan-Level Group Validation
                           │
                           ▼
                 Final Model Evaluation
                           │
                           ▼
                ROC / Confusion Matrices
```

---

# 🔄 Project Development & Script Evolution

Instead of treating every script as an isolated experiment, the scripts represent the **evolution of the project**. Each update was introduced to solve a limitation discovered in the previous stage.

### 1️⃣ Data & ROI Preparation

The initial scripts were created to identify positive nodules, locate the corresponding image regions, and extract ROIs from the LUNA16 data.

### 2️⃣ Dataset Construction & Refinement

The dataset-building scripts were updated to make ROI samples consistent and convert them into a structured dataset suitable for machine learning.

### 3️⃣ Feature Selection

Decision Tree feature importance was introduced to determine which intensity features contributed most to classification rather than automatically using every available feature.

### 4️⃣ Model Comparison

Several classical ML models were added so that Logistic Regression, KNN, SVM, and Random Forest could be evaluated under the same feature representation.

### 5️⃣ Cross-Validation

The original single train/test experiment was extended to **5-fold cross-validation** to obtain more stable estimates and reduce dependence on one random split.

### 6️⃣ Leakage-Free Feature Selection

Feature selection was moved **inside the training fold**. This prevents the test folds from influencing which features are selected.

### 7️⃣ Scan-Level Leakage Investigation

A dedicated analysis was introduced after recognizing that multiple samples could originate from the same CT scan.

The investigation found:

- **200 samples**
- **102 unique Scan IDs**
- **30 scans with multiple samples**
- **4 scans containing both positive and negative samples**

This means a random sample-level split can place correlated samples from the same scan in both training and testing sets.

### 8️⃣ Scan-Level Validation

The final validation stage therefore uses **GroupKFold with Scan ID as the grouping variable**.

This guarantees:

```text
Training Scan IDs ∩ Testing Scan IDs = ∅
```

Feature selection is also performed independently inside every training fold.

### 9️⃣ Final Evaluation & Visualization

The final scripts generate model metrics, ROC curves, confusion matrices, and comparison plots.

> 🧠 **In short:** the project evolved from a basic ROI classification experiment into a more rigorous pipeline with cross-validation, fold-wise feature selection, scan-level leakage prevention, and reproducible model comparison.

---

# 📁 Project Structure

```text
.
├── 00-find_positive_nodule.py
├── 01-create_ROI.py
├── 02-build_dataset.py
├── 03-build_dataset_update.py
├── 04-feature_selection_by_decision_tree.py
├── 05-feature_selection_by_decision_tree_analysis.py
├── 06-feature_selection_cv.py
├── 07-compare_models.py
├── 08-compare_models_by_cross_validation.py
├── 09-nested_feature_selection.py
├── 10-final_evaluation.py
├── 11-check_patient_leakage.py
├── 12-scan_level_leakage_free.py
├── README.md
└── results/
```

> The exact location of generated CSV/PNG outputs depends on the local execution environment.

---

# 🧩 Feature Engineering

The current dataset uses **seven first-order intensity features** extracted from each ROI:

| Feature | Description |
|---|---|
| `Mean` | Average pixel intensity |
| `Median` | Middle pixel intensity |
| `Std` | Standard deviation of intensity |
| `Min` | Minimum intensity |
| `Max` | Maximum intensity |
| `Range` | Difference between maximum and minimum |
| `Variance` | Intensity variance |

These are relatively simple and interpretable features.

### 📌 Important limitation

The current pipeline does **not** use advanced radiomic texture features, deep-learning embeddings, or detailed morphological/shape descriptors.

This makes the project useful as a transparent classical-ML baseline, while also leaving room for future improvement.

---

# 🔎 Feature Selection

A Decision Tree was used to estimate feature importance.

Across the final scan-level cross-validation folds, the average importance was:

| Feature | Mean Importance | Std |
|---|---:|---:|
| Mean | 0.4127 | 0.2304 |
| Variance | 0.3244 | 0.3013 |
| Median | 0.0869 | 0.0600 |
| Max | 0.0692 | 0.0284 |
| Min | 0.0621 | 0.0671 |
| Range | 0.0370 | 0.0542 |
| Std | 0.0078 | 0.0174 |

The most informative features were generally:

- `Mean`
- `Variance`
- `Median`
- `Max`
- `Min`

However, the selected top features varied between folds. This is another reason why feature selection should be performed **inside each training fold** rather than once on the entire dataset.

---

# 🤖 Machine-Learning Models

Four classical classifiers were evaluated:

### 📈 Logistic Regression

A simple and interpretable linear baseline.

### 📍 K-Nearest Neighbors (KNN)

A distance-based classifier that provides a useful non-parametric comparison.

### ⚙️ Support Vector Machine (SVM)

A strong classical classifier for relatively small feature spaces.

### 🌲 Random Forest

An ensemble of decision trees capable of modeling nonlinear relationships and providing feature importance.

---

# 🔬 Validation Strategy

One of the most important parts of this project is the evolution of the validation methodology.

## ⚠️ Why sample-level splitting is risky

At first, samples were evaluated using ordinary sample-level train/test splits and cross-validation.

However:

```text
200 samples
      ↓
102 unique scans
```

Therefore, several samples belong to the same CT scan.

If samples from the same scan appear in both training and testing:

```text
Training
 ├── Scan A / Sample 1
 └── Scan B / Sample 1

Testing
 ├── Scan A / Sample 2   ❌
```

the model may benefit from scan-specific characteristics rather than learning generalizable nodule patterns.

This is a form of **data leakage**.

---

# 🛡️ Leakage Analysis

The dedicated leakage analysis showed:

| Property | Result |
|---|---:|
| Total samples | 200 |
| Unique Scan IDs | 102 |
| Scans with multiple samples | 30 |
| Scans containing both classes | 4 |
| Missing Scan IDs | 0 |

The presence of multiple samples from the same scan makes scan-level grouping essential for a more reliable evaluation.

---

# 🔐 Final Leakage-Free Evaluation

The final evaluation uses:

### `GroupKFold`

with:

```text
Group = Scan_ID
```

For every fold:

- Training and testing Scan IDs are disjoint
- Feature selection is performed using training data only
- The selected features are then evaluated on the unseen test scans

The final pipeline therefore follows:

```text
Fold
 │
 ├── Training scans
 │      │
 │      ├── Feature selection
 │      ├── Model training
 │      └── Model fitting
 │
 └── Testing scans
        │
        └── Final evaluation
```

### ✅ Leakage check

For every fold:

```text
Common Scan IDs = 0
Leakage Check = PASSED
```

This scan-level evaluation should be considered the **primary scientific result** of the project.

---

# 📊 Results

## 🏆 Final Scan-Level Cross-Validation Results

The final leakage-free evaluation produced the following 5-fold results:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| KNN | 0.685 ± 0.055 | 0.670 ± 0.067 | 0.766 ± 0.065 |
| Logistic Regression | 0.735 ± 0.068 | 0.731 ± 0.083 | **0.854 ± 0.072** |
| Random Forest | 0.690 ± 0.055 | 0.681 ± 0.054 | 0.779 ± 0.070 |
| SVM | **0.755 ± 0.045** | **0.732 ± 0.061** | 0.812 ± 0.084 |

### 🥇 Best ROC-AUC

**Logistic Regression**

```text
ROC-AUC = 0.854 ± 0.072
Accuracy = 0.735 ± 0.068
F1-score = 0.731 ± 0.083
```

### 🥇 Best Accuracy

**SVM**

```text
Accuracy = 0.755 ± 0.045
F1-score = 0.732 ± 0.061
ROC-AUC = 0.812 ± 0.084
```

### 🧠 Interpretation

There is no single universally "best" model because the ranking depends on the evaluation metric.

- **Logistic Regression** achieved the highest mean ROC-AUC.
- **SVM** achieved the highest mean accuracy and a very similar F1-score.
- KNN and Random Forest performed lower on the leakage-free scan-level evaluation.

For this project, **ROC-AUC is used as the main criterion when identifying the strongest overall ranking model**, making Logistic Regression the primary model based on the final cross-validation results.

---

# 📈 Earlier Sample-Level Experiments

Before scan-level leakage was identified and corrected, sample-level experiments produced higher or different performance estimates.

For example, in the initial single train/test comparison:

| Feature Set | Model | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|
| All | Logistic Regression | 0.800 | 0.800 | 0.908 |
| All | KNN | 0.750 | 0.750 | 0.821 |
| All | SVM | 0.825 | 0.811 | 0.893 |
| All | Random Forest | 0.750 | 0.762 | 0.826 |
| Top 3 | Logistic Regression | 0.825 | 0.821 | **0.928** |
| Top 3 | KNN | 0.750 | 0.737 | 0.811 |
| Top 3 | SVM | **0.875** | **0.878** | 0.923 |
| Top 3 | Random Forest | 0.725 | 0.744 | 0.820 |

These results were useful during development, but they should **not** be treated as the primary estimate of generalization because the sample-level strategy does not guarantee separation of samples originating from the same CT scan.

This difference is one of the most important findings of the project:

> ⚠️ **A higher validation score does not necessarily mean a better medical ML model if the validation design allows correlated samples to cross the train/test boundary.**

---

# 🧪 Final Evaluation on the Full Sample Set

An additional final evaluation script produced the following sample-level results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.745 | 0.748 | 0.740 | 0.744 | 0.843 |
| KNN | 0.740 | 0.767 | 0.690 | 0.726 | 0.786 |
| SVM | **0.785** | **0.820** | 0.730 | 0.773 | 0.833 |
| Random Forest | **0.785** | 0.813 | **0.740** | **0.775** | 0.838 |

Confusion matrices from this evaluation were:

```text
Logistic Regression
[[75, 25],
 [26, 74]]

KNN
[[79, 21],
 [31, 69]]

SVM
[[84, 16],
 [27, 73]]

Random Forest
[[83, 17],
 [26, 74]]
```

Again, because this is a sample-level evaluation, it is best interpreted as a **development/reference experiment**, not as the main leakage-free generalization result.

---

# 📊 Generated Visualizations

The project generates several visual outputs, including:

- 📈 ROC curves
- 🔲 Confusion matrices
- 📊 Accuracy comparison
- 📊 F1-score comparison
- 📊 ROC-AUC comparison
- 🌳 Feature-importance analysis

Example output files include:

```text
model_comparison.csv
cross_validation_results.csv
final_model_results.csv
roc_curves.png
accuracy_comparison.png
f1_comparison.png
auc_comparison.png
```

---

# 🚀 How to Run the Project

## 1️⃣ Prepare the LUNA16 Dataset

Download the official LUNA16 data and annotations from the official source.

Place the required dataset files in the appropriate local directories used by the scripts.

---

## 2️⃣ Run the Pipeline in Order

The recommended development order is:

```text
00 → 01 → 02/03 → 04/05 → 06 → 07 → 08 → 09 → 11 → 12 → 10
```

A simplified interpretation is:

```text
Raw data
   ↓
Positive nodule identification
   ↓
ROI extraction
   ↓
Dataset construction
   ↓
Feature selection
   ↓
Model comparison
   ↓
Cross-validation
   ↓
Leakage investigation
   ↓
Scan-level leakage-free validation
   ↓
Final evaluation
```

> 💡 Some scripts are exploratory or represent earlier versions of the pipeline. They are kept in the repository to preserve the project's development history.

---

# 🧰 Requirements

The project uses Python and common scientific machine-learning libraries.

Typical dependencies include:

```text
Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
```

Depending on the ROI/image preparation stage, additional medical-image or image-processing packages may be required by the local implementation.

A typical environment can be created with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

If additional packages are required by a specific script, install them before running that stage.

---

# 🧪 Reproducibility

For reproducible experiments:

- Keep the same dataset version.
- Keep Scan IDs attached to every sample.
- Do not perform feature selection on the full dataset before cross-validation.
- Use the same random seeds where specified.
- Keep training and testing scans completely separated.
- Report mean ± standard deviation across folds.

The most important reproducibility rule is:

```text
Never allow the same Scan ID to appear in both training and testing data.
```

---

# ⚠️ Limitations

This project is an experimental classical-ML pipeline and has several limitations.

### 1. Small derived sample set

The final feature dataset contains only:

```text
200 samples
102 unique scans
```

Therefore, results should not be interpreted as definitive clinical performance.

### 2. Simple features

Only seven first-order intensity statistics are currently used.

More informative radiomic features may capture:

- texture
- shape
- heterogeneity
- spatial structure
- morphological characteristics

### 3. Classical ML only

No CNN, 3D deep-learning model, transformer, or learned image representation is included in the current pipeline.

### 4. Scan-level grouping

The current leakage-prevention strategy groups by **Scan ID**. This is appropriate for preventing multiple samples from the same CT scan from crossing the split, but it should not automatically be interpreted as a complete independent-patient validation unless patient identity is explicitly available and grouped.

### 5. Limited dataset scale

The project uses a relatively small derived feature table compared with the full LUNA16 dataset.

---

# 🔮 Future Improvements

Possible future extensions include:

### 🧬 Advanced Radiomics

Add:

- GLCM texture features
- GLRLM features
- Local Binary Patterns
- Shape descriptors
- Higher-order statistical features

### 🧠 Deep Learning

Compare classical ML against:

- 2D CNN
- 3D CNN
- Transfer learning
- Vision Transformers

### 🩻 3D ROI Analysis

Instead of using a single 2D ROI, use volumetric neighborhoods around candidate nodules.

### ⚖️ Better Group-Aware Splitting

Investigate stratified group-aware strategies when appropriate, especially for larger datasets.

### 📊 External Validation

Evaluate the final model on an independent dataset to estimate real-world generalization.

### 🔬 Hyperparameter Optimization

Use nested, group-aware cross-validation for systematic hyperparameter tuning without leakage.

---

# 💡 Key Lessons from the Project

This project demonstrates several important lessons in medical machine learning:

### 1️⃣ More samples do not always mean more independent information

```text
200 samples ≠ 200 independent observations
```

because those samples came from only 102 CT scans.

### 2️⃣ Data leakage can produce overly optimistic results

A model can appear strong simply because related samples from the same scan appear in both training and testing.

### 3️⃣ Feature selection is part of the learning process

If feature selection uses the entire dataset before cross-validation, information from the test folds can indirectly influence the model.

### 4️⃣ Validation strategy matters as much as model choice

A simpler model evaluated correctly can be more scientifically meaningful than a stronger model evaluated with leakage.

### 5️⃣ Medical ML requires careful experimental design

Accuracy alone is not enough. The data structure, independence assumptions, validation strategy, and leakage controls must also be considered.

---

# 🏁 Final Takeaway

The main achievement of this project is not simply obtaining a high classification score.

The most important outcome was the **progression from a basic sample-level experiment to a leakage-aware scan-level machine-learning pipeline**.

The final methodology ensures:

```text
✓ Feature selection inside training folds
✓ Scan-level grouping
✓ No shared Scan IDs between train/test
✓ 5-fold cross-validation
✓ Multiple model comparison
✓ ROC-AUC / F1 / Accuracy reporting
✓ Reproducible evaluation
```

The final scan-level results show that:

> 🏆 **Logistic Regression achieved the highest mean ROC-AUC: 0.854 ± 0.072**

while:

> 🏆 **SVM achieved the highest mean Accuracy: 0.755 ± 0.045**

These results provide a transparent classical-ML baseline for the current ROI representation.

---

# 📚 References

1. LUNA16 Challenge  
   https://luna16.grand-challenge.org/

2. LUNA16 Data  
   https://luna16.grand-challenge.org/Data/

3. LUNA16 Procedure  
   https://luna16.grand-challenge.org/Procedure/

4. LUNA16 Description  
   https://luna16.grand-challenge.org/Description/

5. The Cancer Imaging Archive — LIDC-IDRI  
   https://www.cancerimagingarchive.net/collection/lidc-idri/

---

# 📜 Citation & Attribution

If you use the LUNA16 dataset, please follow the citation and attribution requirements provided by the official LUNA16 and LIDC-IDRI resources.

This repository is an educational/research project and is **not a clinical diagnostic system**.

---

# ❤️ Acknowledgments

Special thanks to the researchers and institutions behind:

- **LIDC-IDRI**
- **The Cancer Imaging Archive (TCIA)**
- **LUNA16 Challenge**
- The open-source Python scientific computing ecosystem

---

## ⭐ If you find this project useful

Feel free to ⭐ **Star** the repository, explore the code, and use the pipeline as a starting point for further research in medical image analysis.

```text
🫁 Medical Imaging
      +
📊 Feature Engineering
      +
🤖 Classical Machine Learning
      +
🛡️ Leakage Prevention
      =
🔬 Reproducible Research
```

