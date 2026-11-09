# Customer Churn Prediction Using Apache Spark

**A PySpark-based data analytics and machine learning project for customer churn prediction using SparkSQL and Spark MLlib.**

## 1. Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. Early identification of at-risk customers can support customer retention strategies and improve business decision-making.

This project implements an end-to-end customer churn prediction workflow using **Apache Spark (PySpark)** on **Google Colab**.

The workflow covers:

- Data ingestion, cleaning, and transformation using Spark DataFrames.
- Exploratory data analysis (EDA) using SparkSQL.
- Feature engineering with Spark ML Pipelines.
- Training and evaluation of multiple classification models.
- Hyperparameter tuning using cross-validation.
- Model comparison, feature importance analysis, and churn prediction on new samples.

The project was developed as an academic Big Data assignment to explore distributed data processing concepts and machine learning workflows.

## 2. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Apache Spark, PySpark, Spark DataFrame |
| Data Analytics | SparkSQL |
| Machine Learning | Spark MLlib |
| Feature Engineering | StringIndexer, OneHotEncoder, VectorAssembler |
| Model Optimization | CrossValidator, ParamGridBuilder |
| Environment | Google Colab, Jupyter Notebook |
| Evaluation | Accuracy, Precision, Recall, F1-score, AUC |

## 3. Dataset

The project uses the **IBM Telco Customer Churn** dataset, containing customer information from a telecommunications service provider.

Each record represents a customer, with attributes covering demographics, subscribed services, account information, billing, and churn status.

**Dataset summary:**

| Property | Description |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Dataset Type | Structured tabular data |
| Records after preprocessing | 7,032 |
| Target Variable | `Churn` |
| Target Classes | `Yes` (1), `No` (0) |
| Train/Test Split | 80% / 20% |

### Key Features

| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has subscribed |
| `Contract` | Contract type |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly service charges |
| `TotalCharges` | Total amount charged |
| `InternetService` | Type of internet service |
| `OnlineSecurity` | Online security subscription |
| `TechSupport` | Technical support subscription |
| `Churn` | Target label indicating customer churn |

Other features include demographic attributes and additional subscribed services.

### Churn Distribution

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,163 | 73.42% |
| Yes | 1,869 | 26.58% |
| **Total** | **7,032** | **100%** |

The dataset has a class imbalance, with churned customers representing approximately 26.6% of all records.

## 4. System Workflow

The system follows a structured data processing and machine learning workflow:

```text
IBM Telco Customer Churn Dataset
              |
              v
       Data Ingestion
       Spark DataFrame
              |
              v
       Data Preprocessing
       - Data type conversion
       - Missing value handling
       - Target label conversion
       - Feature selection
              |
              v
   Exploratory Data Analysis
          SparkSQL
              |
              v
       Feature Engineering
       - StringIndexer
       - OneHotEncoder
       - VectorAssembler
              |
              v
        Train/Test Split
             80/20
              |
              v
         Model Training
          Spark MLlib
              |
              v
       Model Evaluation
       Accuracy / F1 / AUC
              |
              v
       Model Comparison
       Hyperparameter Tuning
              |
              v
       Churn Prediction
```

## 5. Data Processing and Analysis

### 5.1 Data Preprocessing

Data preprocessing is performed using Spark DataFrames.

The main steps include:

1. Converting `TotalCharges` to a numerical data type.
2. Handling missing or invalid values in relevant columns.
3. Converting the `Churn` label from `Yes/No` to `1/0`.
4. Removing `customerID` from the predictive features.
5. Separating categorical and numerical features.

These transformations prepare the dataset for exploratory analysis and machine learning.

### 5.2 Exploratory Data Analysis with SparkSQL

SparkSQL is used to investigate customer churn patterns and relationships between customer attributes.

The analysis covers:

- Overall churn distribution.
- Churn rate by contract type.
- Churn rate by payment method.
- Churn rate by customer tenure.
- Monthly charges across churn and non-churn groups.

**Key observations:**

- Customers on month-to-month contracts show higher churn rates than customers on longer-term contracts.
- Customers with shorter tenure are more likely to churn.
- Customer tenure and contract type are important indicators associated with churn behavior.

### 5.3 Feature Engineering

The project uses a Spark ML feature transformation pipeline with three primary components.

| Component | Purpose |
|---|---|
| `StringIndexer` | Convert categorical values into numerical indices |
| `OneHotEncoder` | Encode categorical features into one-hot vectors |
| `VectorAssembler` | Combine numerical and encoded features into a single feature vector |

Using a structured Spark ML Pipeline helps maintain consistent feature transformations across model training, testing, and prediction.

## 6. Machine Learning Models

Eight classification algorithms were evaluated using Spark MLlib:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient-Boosted Trees (GBT)
5. Multilayer Perceptron (MLP)
6. Naive Bayes
7. Linear Support Vector Classifier (Linear SVC)
8. One-vs-Rest

Additional experiments were performed using hyperparameter-tuned versions of Logistic Regression and Random Forest.

### Hyperparameter Tuning

**Random Forest**

- `numTrees`: [100, 200]
- `maxDepth`: [5, 10, 15]

**Logistic Regression**

- `regParam`: [0.01, 0.1, 0.5]
- `elasticNetParam`: [0.0, 0.5, 1.0]

Cross-validation was used to select model configurations from these parameter grids.

## 7. Model Evaluation

Classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Area Under the ROC Curve (AUC)

**AUC was prioritized for overall model comparison**, as it measures the ability to distinguish between churn and non-churn customers across classification thresholds.

F1-score was also considered to assess classification performance at a particular decision threshold.

### Experimental Results

| Model | Accuracy | F1-score | AUC |
|---|---:|---:|---:|
| Random Forest | 0.8078 | 0.7953 | **0.8583** |
| Logistic Regression | **0.8107** | **0.8058** | 0.8581 |
| Tuned Logistic Regression | 0.8078 | 0.8006 | 0.8557 |
| Tuned Random Forest | 0.8033 | 0.7938 | 0.8555 |
| One-vs-Rest | 0.8100 | 0.8054 | 0.8544 |
| Linear SVC | **0.8107** | 0.8039 | 0.8527 |
| GBT Classifier | 0.8003 | 0.7939 | 0.8436 |
| Multilayer Perceptron | 0.7623 | 0.7444 | 0.8009 |
| Decision Tree | 0.7906 | 0.7810 | 0.7509 |
| Naive Bayes | 0.6788 | 0.6986 | 0.5861 |

*Results obtained from the project's held-out test dataset. Values are rounded to four decimal places.*

### Key Findings

**Random Forest — Highest AUC**

- AUC: **0.8583**
- Demonstrated the strongest overall class discrimination based on AUC.
- Suitable for ranking customers by predicted churn risk within this experiment.

**Logistic Regression — Highest F1-score**

- F1-score: **0.8058**
- Achieved the highest reported Precision and F1-score.
- Provided a strong, relatively interpretable baseline.

**Hyperparameter Tuning**

- Neither tuned Random Forest nor tuned Logistic Regression outperformed its original configuration on the held-out test dataset.
- These results highlight the importance of validating tuning outcomes rather than assuming that increased model complexity always improves performance.

## 8. Feature Importance Analysis

Feature importance analysis was performed to investigate which customer attributes contributed most to churn prediction.

### Random Forest

The main influential features included:

- Customer tenure.
- Contract type.
- Total charges.
- Monthly charges.
- Internet service type.

### Logistic Regression

Model coefficients were analyzed to identify positive and negative associations with churn predictions.

The results indicated that:

- Month-to-month contracts were associated with higher predicted churn risk.
- Longer-term contracts were associated with lower predicted churn risk.
- Shorter customer tenure was an important indicator of churn.

These findings provide analytical insights that may support customer segmentation and retention planning.

## 9. Prediction on New Customer Samples

The trained models were also evaluated using two illustrative customer profiles.

| Customer | Random Forest Churn Probability | Logistic Regression Churn Probability |
|---|---:|---:|
| Customer 01 | 75.12% | 80.40% |
| Customer 02 | 1.67% | 0.98% |

**Customer 01:** A customer with short tenure and a month-to-month contract. Both models predicted a higher probability of churn.

**Customer 02:** A long-tenure customer with a two-year contract. Both models predicted a low probability of churn.

These examples demonstrate how model predictions can support the identification of customers who may require retention attention.

## 10. Running the Project

The project was developed and executed in **Google Colab**.

### Requirements

- Google account for Google Colab.
- Python.
- PySpark and its required runtime dependencies.
- IBM Telco Customer Churn dataset.

### Instructions

1. Download or clone the repository.
2. Open the project notebook in Google Colab.
3. Execute the environment setup and SparkSession initialization cells.
4. Download the IBM Telco Customer Churn dataset.
5. Update the dataset path in the notebook as necessary.
6. Run the notebook cells sequentially.

The notebook covers data ingestion, preprocessing, SparkSQL analysis, feature engineering, model training, evaluation, and prediction.

**Note:** This is a notebook-based academic project. No public production API or deployed web application is included.

## 11. Project Limitations

The current implementation has several limitations:

- **Dataset size:** The IBM Telco dataset is relatively small compared with real-world enterprise Big Data workloads. The project demonstrates Spark processing workflows but does not benchmark distributed performance at enterprise scale.
- **Execution environment:** Experiments were conducted on Google Colab, subject to available computing resources and execution limits.
- **Hyperparameter tuning:** Only a limited parameter search space was explored for Random Forest and Logistic Regression.
- **Class imbalance:** Churned customers form a minority class; additional class-specific evaluation and imbalance-handling strategies could improve the analysis.
- **Generalization:** Results on this dataset do not guarantee equivalent performance on other customer populations.
- **Deployment:** The workflow focuses on offline experimentation rather than production model serving or monitoring.

## 12. Future Improvements

Possible future enhancements include:

- Evaluating larger and more complex datasets.
- Expanding hyperparameter search spaces.
- Applying class imbalance handling techniques.
- Analyzing churn-class recall, precision, and confusion matrices in greater detail.
- Exploring additional deep learning approaches for tabular data.
- Developing an API or dashboard for customer churn risk monitoring.

## 13. Project Information

**Project Type:** Academic Group Project

**Institution:** VNUHCM – University of Information Technology

**Course:** Big Data Technology

This repository documents the implementation and experimental findings of a customer churn prediction workflow using PySpark, SparkSQL, and Spark MLlib.
