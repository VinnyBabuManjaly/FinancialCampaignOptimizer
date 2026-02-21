# Bank Deposit Subscription Predictor

## Overview

**Bank Deposit Subscription Prediction** applies the CRISP-DM framework to compare KNN, Logistic Regression, Decision Trees, and SVM on the UCI Bank Marketing dataset (41,188 Portuguese telemarketing campaigns, 2008-2010). 

Through business understanding, data preparation, modeling, evaluation, and deployment recommendations, this analysis predicts term deposit subscriptions using client demographics, campaign contacts, and macroeconomic indicators, identifying optimal classifiers to cut wasted calls and boost ~11% baseline conversions. 

The Jupyter Notebook delivers actionable insights for targeted banking outreach.

## 1. Business Understanding

### 1.1 Business Objectives

**Background:**
A Portuguese bank conducted telemarketing campaigns between 2008 and 2010 to promote term deposit subscriptions. Over 41,000 calls were made, but the overall conversion rate was only about 11%. This highlights a clear need to improve targeting and reduce inefficient outreach.

**Primary Objective:**
The goal is to build a predictive model that can identify which clients are most likely to subscribe to a term deposit. The model will use customer demographics, past campaign interactions, and macroeconomic indicators to help the bank focus on high-potential clients instead of relying on mass calling.

**Key Questions:**

* Which customer segments (based on age, occupation, or economic conditions) are most likely to subscribe?
* How do factors like contact timing and previous campaign outcomes affect the likelihood of success?

**Success Criteria:**
Select the best-performing classifier (KNN, Logistic Regression, Decision Trees, or SVM) that reduces unnecessary calls by 20–30% while maintaining more than 85% prediction accuracy on holdout data.

### 1.2 Assess the Situation

This project uses the publicly available UCI Bank Marketing dataset, which contains over 41,000 anonymized records from Portuguese telemarketing campaigns. Since this is a research-based analysis, no live bank data is accessed. The models are developed and compared offline using standard Python tools.

**Requirements & Assumptions:**
The models must be interpretable so that business stakeholders can understand and trust the results. The dataset is legally available for research purposes. We assume that patterns observed in the 2008–2010 campaigns are still relevant for modern marketing strategies. The “duration” feature is excluded to ensure realistic predictions, as it would not be known before making a call.

**Constraints:**
The dataset includes only 20 features, which limits the scope of analysis. There is no real-time deployment involved, and the focus is strictly on evaluating four classifiers: KNN, Logistic Regression, Decision Trees, and SVM.

**Risks & Contingencies:**
Class imbalance could affect model performance, since most clients did not subscribe. Cross-validation will be used to reduce the risk of poor generalization. Another limitation is that the economic context from 2008–2010 may not fully reflect current conditions, which should be acknowledged in any recommendations.

**Costs & Benefits:**
The project has minimal cost because it relies on open-source tools and publicly available data. The potential benefit is significant: reducing wasted calls by 20–30% from the current ~89% non-conversion rate, which could meaningfully improve the bank’s marketing efficiency and return on investment.

**Input variables**(Data already provided)

Index | Variable | Type | Description
------|----------|------|--------------------------------------------
1 | age | numeric | Client's age in years
2 | job | categorical | Type of job (admin., blue-collar, entrepreneur, etc.)
3 | marital | categorical | Marital status (divorced, married, single, unknown)
4 | education | categorical | Education level (basic.4y, high.school, university.degree, etc.)
5 | default | categorical | Has credit in default? (no, yes, unknown)
6 | housing | categorical | Has housing loan? (no, yes, unknown)
7 | loan | categorical | Has personal loan? (no, yes, unknown)
8 | contact | categorical | Contact communication type (cellular, telephone)
9 | month | categorical | Last contact month (jan, feb, mar, ..., dec)
10 | day_of_week | categorical | Last contact day (mon, tue, wed, thu, fri)
11 | duration | numeric | Last contact duration in seconds (benchmark only)
12 | campaign | numeric | Contacts during this campaign (includes last contact)
13 | pdays | numeric | Days since last contact (999 = not previously contacted)
14 | previous | numeric | Contacts before this campaign
15 | poutcome | categorical | Previous campaign outcome (failure, nonexistent, success)
16 | emp.var.rate | numeric | Employment variation rate - quarterly indicator
17 | cons.price.idx | numeric | Consumer price index - monthly indicator
18 | cons.conf.idx | numeric | Consumer confidence index - monthly indicator
19 | euribor3m | numeric | Euribor 3 month rate - daily indicator
20 | nr.employed | numeric | Number of employees - quarterly indicator
21 | y | binary | Target: subscribed term deposit? (yes, no)

### 1.3 Data Mining Goals

**Data Mining Goals:**
Develop and compare four binary classification models—KNN, Logistic Regression, Decision Trees, and SVM—to predict whether a client will subscribe to a term deposit. The models will use the available client, campaign, and economic features.

**Success Criteria:**
Identify the best-performing model with at least 85% accuracy and an AUC-ROC of 0.45 or higher using stratified cross-validation. The model should also demonstrate a meaningful lift over the baseline conversion rate and provide clear outputs such as feature importance and a confusion matrix to support business interpretation.

### 1.4 Project Plan

**Stages:**

1. **Data Understanding:** Load and explore the dataset, review feature distributions, and assess class imbalance.

2. **Data Preparation:** Clean and encode categorical variables, remove the “duration” feature, and split the data into training and test sets.

3. **Modeling:** Train and tune KNN, Logistic Regression, Decision Tree, and SVM models.

4. **Evaluation:** Compare models using cross-validation metrics such as accuracy, AUC, and F1 score, and review feature importance where applicable.

5. **Reporting:** Summarize findings and recommendations in the final notebook.

**Tools:** Jupyter Notebook, pandas, and scikit-learn.

**Risks & Mitigation:**
Address class imbalance with stratified sampling. If certain models struggle to converge, use Logistic Regression as a stable baseline.

**Review Point:**
After initial evaluation, confirm that the top model meets performance expectations; refine data preparation if necessary.

## 2. Data Understanding

### 2.1 Collect Initial Data

**Dataset Acquired**: UCI Bank Marketing dataset (`bank-additional-full.csv`, 41,188 records)

**Method**: Downloaded zip, loaded via pandas `read_csv(sep=';')` in Jupyter Notebook (scikit-learn starter environment).

**Issues**: None encountered. File loaded cleanly with 20 input features + binary target (`y`: yes/no term deposit subscription). Ready for exploration.

### 2.2 Describe Data

* **Dataset:** 41,188 records × 21 columns (20 features + 1 target `y`)
* **Feature Types:** 11 categorical and 10 numeric variables
* **Target Variable:** Binary outcome (`yes` / `no` for term deposit subscription)
* **Data Quality:** No missing values; data loaded cleanly
* **Duplicates:** 12 duplicate records identified (to be removed)
* **Class Imbalance:** ~11% positive class (imbalanced dataset)
* **“Unknown” Labels:** Present in some categorical fields; treated as separate categories
* **Numeric Features:** No major outliers requiring immediate treatment
* **Conclusion:** Dataset is clean, structured, and suitable for classifier comparison

### 2.3 Explore Data

The dataset is highly imbalanced, with only about 11% of clients subscribing to a term deposit. This confirms that predicting the positive class will be challenging and that accuracy alone may be misleading.

Age does not appear to be a strong differentiator. Subscribers are only slightly older on average (around 41 vs. 40 years), though their age distribution is more spread out.

A major insight is that 96% of clients were never previously contacted (`pdays = 999`). This means prior campaign history is limited for most customers. However, when a previous campaign was successful, the subscription rate jumps significantly to about 65%, making it the strongest predictive signal in the data.

Looking at job categories, students and retirees show notably higher subscription rates compared to blue-collar or services roles. In terms of education, university graduates convert more often than those with basic education levels.

Timing also matters. Subscription rates are much higher in months like March, April, September, October, and December, while May and summer months show lower success rates.

From a numeric perspective, macroeconomic variables such as employment variation rate, Euribor rate, and number of employees are highly correlated with each other, indicating strong economic clustering effects.

Overall, prior campaign success, certain job categories (student, retired), higher education levels, and campaign timing appear to be the most meaningful drivers of subscription behavior.


## 3. Data Preparation

### 3.1 Select Data

- Removed 12 duplicate records to prevent any bias in model training.

- Excluded the `duration` feature since it is only known after a call is completed and would introduce data leakage.

- Retained all other features and rows, as there are no true missing values. The “unknown” entries are valid categories and remain useful for modeling.

### 3.2 Clean Data

- The dataset contains a balanced mix of categorical and numeric features. 

- Most variables have no true missing values. 

- Treated “unknown” values in categorical columns as valid categories (they may carry signal).

- Separated numeric and categorical features.

- All categorical features were one-hot encoded, and numeric features were passed through without scaling at this stage. After preprocessing, the dataset expanded from the original features to 52 total input variables due to encoding.

- Overall, the data remains largely complete and usable

### 3.3 Construct Data

#### Key Findings

- Created three new features to improve interpretability and modeling:

  - **age_group** to segment clients into young, middle, and senior categories.
  - **high_campaign** to flag aggressive outreach (4 or more contacts).
  - **econ_stress** to summarize overall economic pressure using macro indicators.

- Addressed class imbalance by oversampling the minority “yes” class (from ~4,600 to ~18,000), improving the distribution to roughly 67% no and 33% yes.

- Although oversampling improves balance, it may not be included in final modeling since many classifiers can handle imbalance using built-in class weighting.

- Overall, these enhancements strengthen business understanding while improving model stability and readiness.

### 3.4 Intergrate data

* Used a stratified 80/20 train–test split to preserve the true 11.3% positive class distribution in both sets.

* Encoded the target variable as binary: `yes → 1`, `no → 0`.

* Applied feature scaling and one-hot encoding using a `ColumnTransformer` pipeline to ensure consistent preprocessing.

* Final prepared datasets:

  * `X_train_prep`: 32,940 × 56
  * `X_test_prep`: 8,236 × 56

* Class imbalance was intentionally preserved to reflect real-world business conditions during evaluation.


## 4.Modelling

### 4.1 Select Modeling Technique


**Selected Techniques**: Four supervised binary classification algorithms as required by assignment:  
1. **K-Nearest Neighbors (KNN)** - Non-parametric, distance-based  
2. **Logistic Regression** - Linear baseline model  
3. **Decision Trees** - Tree-based partitioning  
4. **Support Vector Machines (SVM)** - Maximum margin hyperplane  

**Modeling Approach**:  
- **Baseline**: Logistic Regression (simplest, interpretable)  
- **Comparators**: KNN, Decision Tree, SVM (default hyperparameters first)  
- **Improvement**: Grid search hyperparameter tuning on top performers  
- **Metrics**: Accuracy + F1-score (due to 11.3% class imbalance) + fit time  

**Key Assumptions**:  
- **Data**: Clean, no missing values (handled in 3.2), scaled numeric features (3.5)  
- **Target**: Binary encoded (yes/no → 1/0), stratified splits preserve 11.3% prevalence  
- **KNN**: Euclidean distance meaningful after StandardScaler normalization  
- **Logistic Regression**: Linear decision boundary reasonable for baseline  
- **Decision Tree**: No normality assumptions, handles categoricals naturally  
- **SVM**: Linearly separable or kernel transformable (RBF default)  

### 4.2 Test Design


**Test Strategy**: Stratified train/test split (80/20) already performed in 3.5 preserves real-world 11.3% class prevalence in both sets (32,940 train, 8,236 test records).

**Evaluation Pipeline**:
1. **Baseline Establishment**: Logistic Regression (default hyperparameters) → accuracy, F1-score, fit time benchmark
2. **Default Model Comparison**: Train all 4 classifiers (KNN, Logistic Regression, Decision Tree, SVM) with defaults → same train/test sets
3. **Hyperparameter Tuning**: GridSearchCV (5-fold stratified cross-validation) on top 2 performers from defaults
4. **Final Evaluation**: Best tuned models on held-out test set

**Performance Metrics** (multi-metric due to imbalance):
- **Primary**: F1-score (balances precision/recall for rare 11.3% positive class)
- **Secondary**: Accuracy, Precision, Recall, AUC-ROC
- **Business**: Fit time (practical deployment consideration)

**Test Scenarios**:
1. **Default hyperparameters** → Raw algorithm performance
2. **Tuned hyperparameters** → Optimization potential  
3. **Feature importance** → Business interpretability (Tree coefficients, LR weights)

### 4.3 Build Model with Default Parameters

* Created a **generic model evaluation function** with metrics and visualizations for consistent performance assessment.

* Tested **baseline dummy classifiers** to establish a reference point for comparison.

* Built **Logistic Regression, KNN, Decision Tree, and Linear SVM** using default parameters for initial benchmarking.

* Observed that **SVM with RBF or full linear kernel** is too slow for real-time deployment in banking scenarios.

* Applied **class_weight='balanced'** where possible to automatically account for the 88.7:11.3 class imbalance.

### 4.4 Assess Model(No Tuning)

Below is a technical assessment of the models based strictly on evaluation metrics. No parameter tuning or refinement is considered at this stage.

**Dummy Baseline**   
Not a usable model. It predicts only the majority class and completely fails to detect subscribers. Serves only as a reference point.

**Logistic Regression**  
Strong overall performer. Best ROC-AUC and good balance between recall and specificity. Captures a majority of positive cases while maintaining reasonable false positives. Technically the most robust and stable model.

**KNN**  
High overall accuracy but poor recall. Misses many positive cases (high false negatives). Not ideal when identifying subscribers is important.

**Decision Tree**   
Moderate accuracy but weaker discriminative ability. Lower AUC and F1 indicate poorer separation between classes compared to other models.

**Linear SVM**   
Performance comparable to Logistic Regression, with similar F1 and recall. However, computational cost is significantly higher, which may limit scalability.

Overall Ranking (Based on Predictive Quality)

1. **Logistic Regression** – Best overall balance (highest ROC-AUC, strong recall, stable performance)
2. **Linear SVM** – Similar predictive power but computationally expensive
3. **KNN** – High accuracy but weak recall
4. **Decision Tree** – Moderate performance
5. **Dummy Baseline** – Not suitable


Logistic Regression currently provides the best trade-off between discrimination ability, recall of positive cases, and overall stability. It outperforms other models on ROC-AUC and balanced accuracy, making it the strongest candidate at this stage of assessment.

### 4.5 Build Model With Tuning

##### Key points

* Developed a **generic model tuning function** to streamline training, evaluation, and comparison across models.

* Built **Logistic Regression, KNN, Decision Tree** with hyper parameter tuning.

* **Linear SVM** was tuned with `C = [0.1, 1, 10]` using a 25% random subsample (~11K rows) and 3-fold GridSearchCV to optimize F1 score.

* Full RBF or linear SVM on all data with 5-fold CV was too slow, confirming that **linear kernel with subsampling** is more practical for this dataset.

### 4.6 Assess Results With Tuning

## 5.Evaluation

### 5.1 Evaluate results

### 5.2 Review process

### 5.3 Next steps

## 6. Deployment

### 6.1 Deployment Strategy

### 6.2 Monitoring and Maintenance

### 6.3 Final Report

### 6.4 Review project
