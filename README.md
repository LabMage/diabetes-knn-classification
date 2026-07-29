# Diagnosing Diseases using kNN

## An Application of kNN to Diagnose Diabetes

**Authors:** Elena Boiko and Jacqueline Razo  
**Advisor:** Dr. Achraf Cohen  
**Course:** IDC 6940 — Capstone Projects in Data Science  
**University:** University of West Florida

[View the Full Project Report](https://labmage.github.io/diabetes-knn-classification/) | [View the Presentation Slides](https://labmage.github.io/diabetes-knn-classification/slides.html#/title-slide)

## About the Project
diabetes-knn-classification 
This capstone investigates whether k-Nearest Neighbors (kNN), despite its simplicity and computational limitations, can effectively classify diabetes and prediabetes in a large, imbalanced healthcare dataset. The analysis uses 253,680 responses from the CDC Behavioral Risk Factor Surveillance System and examines how data cleaning, feature scaling, class balancing, feature selection, distance metrics, and neighbor weighting affect model performance.

The project goes beyond reporting overall accuracy, which can be misleading when diabetic cases represent a minority of the data. Four kNN configurations were evaluated and compared with Decision Tree and Random Forest models using ROC-AUC, precision, recall, and F1 score. The strongest kNN configuration combined StandardScaler, SMOTE, feature selection, Euclidean distance, distance weighting, and k = 15. It achieved a ROC-AUC of 0.88 and identified 88% of diabetes and prediabetes cases, outperforming the comparison models in minority-class detection.

This work demonstrates how thoughtful preprocessing and evaluation can transform a basic algorithm into a strong classifier for an imbalanced healthcare problem. It also reflects the project’s central priority: selecting a model based not only on overall accuracy, but on its ability to recognize the cases that are easiest—and potentially most consequential—to miss.

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

The strongest kNN model was then compared with Decision Tree and Random Forest classifiers.

### Best kNN Configuration

- **Neighbors:** k = 15
- **Distance metric:** Euclidean
- **Weighting:** Distance-based
- **Scaler:** StandardScaler
- **Class balancing:** SMOTE
- **Feature selection:** Applied

## Results

| Model | Accuracy | ROC-AUC | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|---:|
| **kNN with SMOTE and feature selection** | **0.78** | **0.88** | **0.73** | **0.88** | **0.80** |
| Decision Tree with SMOTE | 0.72 | 0.80 | 0.70 | 0.78 | 0.74 |
| Decision Tree without SMOTE | 0.86 | 0.81 | 0.52 | 0.15 | 0.24 |
| Random Forest without SMOTE | 0.87 | 0.82 | 0.59 | 0.13 | 0.21 |

Although the unbalanced Decision Tree and Random Forest models produced higher overall accuracy, their recall for the diabetes and prediabetes class was very low. This means they failed to identify most minority-class cases.

The optimized kNN model provided the strongest balance between precision and recall. It achieved the highest ROC-AUC and minority-class F1 score while identifying 88% of diabetes and prediabetes observations.

## Key Findings

- Overall accuracy alone was not sufficient for evaluating this imbalanced healthcare dataset.
- Models trained without class balancing favored the majority class and missed many diabetes and prediabetes cases.
- SMOTE substantially improved minority-class detection.
- Scaling was essential because kNN calculates distances between observations.
- Feature selection and distance weighting improved kNN performance.
- The optimized kNN model outperformed the comparison models in ROC-AUC, recall, and F1 score for the minority class.

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
- `knn_tuning_results.csv` — kNN tuning results
- `images/` and `slides_files/` — figures and presentation resources

## Project Attribution

This project was completed collaboratively as part of the M.S. Data Science capstone course at the University of West Florida. This repository and its published GitHub Pages site are maintained by Elena Boiko.

## Important Note

This project was created for academic and educational purposes. The model is not a clinical diagnostic system, and its predictions should not be used as a substitute for professional medical evaluation.