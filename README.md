# Bank Marketing Conversion Prediction Using Machine Learning

**Author:** Heppy

## 1. Project Overview
This project develops an explainable, leakage-aware machine-learning solution for predicting whether a bank customer will subscribe to a term deposit. It treats prediction as a customer-prioritisation and decision-support problem rather than only a classification task.

## 2. Research Aim
To design, implement, and critically evaluate an explainable machine-learning artefact that predicts term-deposit subscription, compares classification models, avoids prediction-time leakage, and supports responsible campaign targeting.

## 3. Research Objectives
- Prepare and analyse the Bank Marketing dataset while separating pre-contact variables from information generated during calls.
- Develop and compare machine-learning classifiers using temporal validation and imbalance-aware evaluation.
- Interpret the selected model using SHAP and related feature-importance methods.
- Evaluate calibration, threshold behaviour, campaign lift, and operational usefulness.
- Produce a reproducible Python prototype for responsible customer prioritisation.

## 4. Dataset
**Dataset name:** `cestwc/bank-marketing`  
**Platform:** Hugging Face Datasets  
**Original source:** UCI Machine Learning Repository – Bank Marketing  
**Task:** Binary classification (`y`: term-deposit subscription — yes/no)  
**Records:** 45,211  
**Predictors:** 16

Dataset: https://huggingface.co/datasets/cestwc/bank-marketing  
Original UCI source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## 5. Technologies Used
- Python 3.12
- Pandas
- NumPy
- scikit-learn 1.6.1
- Hugging Face `datasets`
- SHAP 0.52.0
- MLflow 3.15.1
- Joblib
- Google Colab / Jupyter Notebook

## 6. Data Preprocessing
- Verified dataset dimensions, target classes, feature ranges, and provenance.
- Decoded Hugging Face `ClassLabel` categorical features.
- Interpreted `pdays = 999` as the Hugging Face replacement for UCI's `-1` sentinel meaning *not previously contacted*.
- Preserved legitimate negative account balances and `unknown` categorical values.
- Separated strict pre-contact features from campaign/post-contact variables.
- Excluded `duration` from the deployable model because it is unavailable before a call and creates prediction-time leakage.
- Used one-hot encoding for categorical variables and scaling where required for Logistic Regression.
- Applied preprocessing inside model pipelines to reduce leakage.
- Used chronological partitions: approximately 70% training, 7.5% calibration, 7.5% threshold selection, and 15% untouched testing.
- Used expanding-window temporal cross-validation during model development.

## 7. Evaluation Metrics
- Average Precision (AP)
- ROC-AUC
- Precision
- Recall
- F1-score / F2-oriented threshold selection
- Balanced Accuracy
- Brier Score
- Log Loss
- Expected Calibration Error (ECE)
- Confusion Matrix
- Cumulative Gains
- Campaign Lift
- SHAP and Permutation Importance

## 8. Experimental Results
Baseline models included Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, Extra Trees, and Histogram Gradient Boosting.

- During expanding-window development validation, **Logistic Regression** achieved AP ≈ **0.0866**, slightly above Random Forest (≈ **0.0863**).
- On the untouched later test period, **Extra Trees** achieved the highest baseline AP (≈ **0.4699**), followed by Histogram Gradient Boosting (≈ **0.4672**), Random Forest (≈ **0.4577**), and Logistic Regression (≈ **0.4375**).
- The difference between development and test results indicated substantial temporal distribution shift.
- Adding call `duration` greatly increased measured discrimination, confirming that it is a strong but operationally invalid leakage feature.

## 9. Performance
The final active model was **Logistic Regression**, selected from the temporal development stage.

| Measure | Result |
|---|---:|
| Test Average Precision | 0.4375 |
| AP Bootstrap 95% CI | 0.4240–0.4532 |
| Test ROC-AUC | 0.5538 |
| ROC-AUC 95% CI | 0.5388–0.5689 |
| Selected threshold | 0.035 |
| Recall at selected threshold | 97.56% |
| Precision at selected threshold | 39.56% |
| F1 at selected threshold | 0.5629 |
| Specificity at selected threshold | 3.33% |

Campaign ranking was more operationally useful than a fixed binary threshold:
- Top 5% customer group: lift ≈ **1.316**
- Top 10%: lift ≈ **1.187**
- Top 20%: lift ≈ **1.128**

## 10. Prototype
The prototype is a reproducible Python-based decision-support workflow that:
1. Loads and validates the dataset.
2. Audits feature availability and prevents leakage.
3. Preprocesses numerical and categorical variables.
4. Trains and compares classification models.
5. Calibrates probabilities and selects decision thresholds.
6. Ranks customers by predicted conversion probability.
7. Generates campaign lift/gains analysis.
8. Produces SHAP and permutation-based explanations.
9. Tracks experiments with MLflow and supports model persistence with Joblib.

> **Deployment note:** The prototype is intended for research and decision support. It is not a production banking CRM, autonomous calling system, or fully deployed marketing platform.

## References
- Hugging Face: `cestwc/bank-marketing`
- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
