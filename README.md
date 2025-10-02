![Plots Screenshot](images/Screenshot%202025-10-02%20165531.png)

# Customer Churn EDA & ML Pipeline 🎯📊

> Project snapshot: Hands-on project to explore, clean, visualize, profile, and frame a machine-learning problem from a customer dataset (churn prediction). This repository contains the dataset, a Jupyter notebook (`data_profiler.ipynb`) and visual outputs created during Exploratory Data Analysis (EDA).

---

## 🚀 Project Overview

This project walks through a complete data workflow from acquisition to problem framing for a binary classification task (predicting `churn`). Students will practice:

- Loading and merging data from multiple sources
- Cleaning and preprocessing real-world data
- Performing Univariate, Bivariate and Multivariate EDA with visualizations
- Generating automated profiling reports for faster decision-making
- Framing a machine learning problem and preparing the dataset for modeling


## 🗂 Dataset

Sample columns (from `customer_data.csv`):

- `customer_id` — Unique identifier
- `name` — Customer name
- `age` — Age (numeric, contains missing values)
- `gender` — Male/Female (categorical)
- `income` — Annual income (numeric)
- `purchases` — Number of purchases (numeric)
- `churn` — Target variable (0 = retained, 1 = churned)

> The loaded file: `customer_data.csv` (and the analysis notebook `data_profiler.ipynb`) — these are included in the project workspace.


## 🎯 Learning Outcomes

By completing this project students will:

- Gain practical experience loading and merging datasets from CSV / JSON / SQL / API. ✅
- Understand the role of tensors/numerical arrays in ML pipelines. 🧠
- Perform data cleaning & preprocessing (imputation, type correction, deduplication). 🧹
- Carry out Univariate, Bivariate & Multivariate EDA with clear visualizations. 📈
- Generate automated profiling reports (for exploratory decision-making). 🧾
- Frame a machine learning problem: define target, candidate features and baseline approach. ⚙️


## 🛠 What was performed (summary of tasks)

### Part B — Data Acquisition
- Loaded CSV with `pandas.read_csv()`.
- Parsed JSON file with `pandas.read_json()`.

### Part C — Data Understanding & Cleaning
- Initial exploration: `df.head()`, `df.info()`, `df.describe()`.
- Missing value checks: `df.isnull().sum()` and visual checks.
- Handling missing data: imputation strategies (mean/median for numeric; mode for categorical) or row removal when appropriate.

### Part D — Exploratory Data Analysis (EDA)
Univariate Analysis
- Distribution plots for `age`, `income`, and `purchases` (histograms + KDE).

Bivariate Analysis
- Relationship between `gender` and `purchases` (boxplots / violin plots).
- Relationship between `income` and `churn` (histograms, boxplots, or grouped statistics).

Multivariate Analysis
- Correlation heatmap of numerical variables to spot collinear features.
- Pair plots to inspect feature interactions and distributions.

_Example plotting snippets:_

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['age'], kde=True)
plt.title('Age distribution')

sns.boxplot(x='gender', y='purchases', data=df)

sns.heatmap(df.corr(), annot=True, fmt='.2f')

sns.pairplot(df.select_dtypes(include=['number']))
```

## 📊 Automated Profiling
- A profiling report (e.g. `pandas_profiling.ProfileReport` or `ydata_profiling`) was generated inside `data_profiler.ipynb` to provide a fast overview of distributions, missingness, correlations, and warnings.

```python
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Customer Data Profile')
profile.to_file('reports/customer_profile.html')
```

Made with ❤️ — open to edits & improvements.
