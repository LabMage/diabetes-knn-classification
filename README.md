# Diagnosing Diseases using kNN

## An Application of kNN to Diagnose Diabetes

**Authors:** Elena Boiko and Jacqueline Razo  
**Advisor:** Dr. Achraf Cohen  
**Course:** IDC 6940 — Capstone Projects in Data Science  
**University:** University of West Florida

[View the Full Project Report](https://labmage.github.io/diabetes-knn-classification/) | [View the Presentation Slides](https://labmage.github.io/diabetes-knn-classification/slides.html#/title-slide)

## About the Project

This capstone investigates whether k-Nearest Neighbors (kNN), despite its simplicity and computational limitations, can effectively classify diabetes and prediabetes in a large, imbalanced healthcare dataset. The analysis uses 253,680 responses from the CDC Behavioral Risk Factor Surveillance System and examines how data cleaning, feature scaling, class balancing, feature selection, distance metrics, and neighbor weighting affect model performance.

The project goes beyond reporting overall accuracy, which can be misleading when diabetic cases represent a minority of the data. Four kNN configurations were evaluated and compared with Decision Tree and Random Forest models using ROC-AUC, precision, recall, and F1 score. After a methodological revision in 2026, all models were evaluated on one common, untouched test set. SMOTE raised kNN recall for diabetes and prediabetes cases from about 0.2 to about 0.6, and the best kNN ROC-AUC was 0.752 (kNN 2). The tree-based models performed better: Random Forest achieved the highest ROC-AUC (0.813), and Decision Tree with SMOTE the highest recall (0.740).

This work shows how preprocessing choices change the behavior of kNN on an imbalanced healthcare problem, and why models should be selected not only on overall accuracy, but on their ability to recognize the cases that are easiest—and potentially most consequential—to miss.

## Dataset

The analysis uses the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

- **Observations:** 253,680 survey responses
- **Predictors:** 21 health, lifestyle, and demographic variables
- **Target:** `Diabetes_binary`
- **Class 0:** No diabetes
- **Class 1:** Diabetes or prediabetes
- **Initial class distribution:** 86.07% class 0 and 13.93% class 1

The predictors include BMI, age, general health, physical and mental health, physical activity, income, education, healthcare access, and medical history.

## Data Preparation

The dataset was examined for missing values, duplicate observations, outliers, class imbalance, and relationships among predictors.

The preprocessing workflow included:

- Removing 24,206 duplicate records
- Retaining ordinal variables in their meaningful ranked form
- Scaling distance-sensitive features
- Comparing StandardScaler and RobustScaler
- Applying SMOTE to address class imbalance
- Using feature selection to reduce dimensional noise
- Preparing consistent training and testing data for model evaluation

After duplicate removal, the minority-class proportion increased from approximately 13.9% to 15.3%.

## Modeling Approach

Four kNN configurations were evaluated by varying:

- Number of neighbors
- Euclidean and Manhattan distance metrics
- Uniform and distance-based weighting
- StandardScaler and RobustScaler
- SMOTE application
- Feature selection

kNN 4 (SMOTE + chi-square feature selection) was then compared with Decision Tree and Random Forest classifiers.

## Results

All models were evaluated on the same untouched test set (68,843 observations, 15.29% class 1).

| Model | Accuracy | ROC-AUC | PR-AUC | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|---:|---:|
| kNN 1 (baseline) | 0.832 | 0.705 | 0.281 | 0.406 | 0.206 | 0.274 |
| kNN 2 (RobustScaler, Manhattan) | 0.841 | 0.752 | 0.334 | 0.446 | 0.155 | 0.230 |
| kNN 3 (SMOTE) | 0.691 | 0.733 | 0.286 | 0.278 | 0.641 | 0.388 |
| kNN 4 (SMOTE + feature selection) | 0.731 | 0.722 | 0.300 | 0.303 | 0.579 | 0.397 |
| Decision Tree | 0.849 | 0.794 | 0.388 | 0.517 | 0.154 | 0.237 |
| Decision Tree with SMOTE | 0.671 | 0.756 | 0.317 | 0.281 | **0.740** | **0.407** |
| **Random Forest** | **0.854** | **0.813** | **0.432** | 0.599 | 0.134 | 0.219 |

Models trained on imbalanced data reached high accuracy but identified only 13–21% of diabetes and prediabetes cases. SMOTE increased recall substantially at the cost of precision. Random Forest separated the classes best (highest ROC-AUC and PR-AUC), while Decision Tree with SMOTE detected the most minority-class cases.

Results are produced by `analysis/run_all_models.py` and saved in `analysis/results.csv`.

## Key Findings

- Overall accuracy alone was not sufficient for evaluating this imbalanced healthcare dataset.
- Models trained without class balancing favored the majority class and missed many diabetes and prediabetes cases.
- SMOTE substantially improved minority-class detection.
- Scaling was essential because kNN calculates distances between observations.
- RobustScaler, Manhattan distance and distance weighting gave the best kNN ROC-AUC (0.752); chi-square feature selection reduced the features from 21 to 12 without improving results.
- Tree-based models outperformed kNN on this dataset: Random Forest had the best ROC-AUC (0.813).

## Technologies Used

- Python
- R
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- ggplot2
- Quarto
- RStudio
- GitHub Pages

## Repository Contents

- `index.qmd` — complete project report and analysis source
- `index.html` — published project report
- `slides.qmd` — presentation source
- `slides.html` — published presentation
- `references.bib` — academic references
- `cdc_data.csv` — project dataset
- `eda.csv` — exploratory analysis data
- `analysis/` — corrected evaluation script, results and ROC curves
- `images/` and `slides_files/` — figures and presentation resources

## Methodological Revision (2026)

The original 2025 evaluation reported ROC-AUC 0.88 for the kNN model with SMOTE and feature selection. In October 2026, Elena Boiko reviewed the code and found data leakage: for this model and for the Decision Tree with SMOTE, SMOTE (and for kNN also feature selection and scaling) had been applied before the train-test split, and the tree models had been evaluated on a different test set that included duplicates. All preprocessing is now fitted on training data only, and all seven models are evaluated on one common test set. The results above replace the original figures.

## Project Attribution

This project was completed collaboratively as part of the M.S. Data Science capstone course at the University of West Florida. This repository and its published GitHub Pages site are maintained by Elena Boiko.

## Important Note

This project was created for academic and educational purposes. The model is not a clinical diagnostic system, and its predictions should not be used as a substitute for professional medical evaluation.