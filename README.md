# Telecom_Analysis

![Telecom Analytics Dashboard](preview.png)

# 📊 Telecom Customer Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-6f42c1.svg)](https://plotly.com/python/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-f7931e.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg)](https://pandas.pydata.org/)

> A Product Data Science–oriented telecom analytics project that transforms telecom usage data into actionable insights across customer overview, engagement, experience, and satisfaction workflows.

---

## 🚀 Project Summary

This project analyzes telecom customer behavior using Python, exploratory data analysis, feature engineering, clustering, and an interactive Streamlit dashboard.

It is designed to communicate the kind of workflow a **Product Data Scientist** would use to:
- understand user behavior,
- profile customer engagement,
- explore experience-related metrics,
- segment users into meaningful groups,
- and present insights through a clean interactive dashboard.

The repository includes:
- a **Streamlit app** for dashboard delivery,
- **data files** for cleaned and intermediate analysis outputs,
- **notebooks** for analysis work,
- and **Python scripts** for cleaning and helper utilities.

---

## 🎯 Product Data Science Angle

This project is positioned as a **product analytics + behavioral segmentation** case study in the telecom domain.

Key product questions this project helps answer:
- Which users are most active and valuable?
- How does usage differ across apps like YouTube, Google, Netflix, Email, Gaming, and Social Media?
- Which users show high engagement versus low engagement?
- How can clustering help identify meaningful behavioral segments?
- How can telecom experience and satisfaction analysis be extended into retention and churn use cases?

---

## 🧩 Core Analysis Modules
### 1. User Overview Analysis
Focuses on broad customer and dataset understanding:
- dataset quality checks
- missing value analysis
- handset manufacturer and handset type exploration
- user/session distribution
- total usage patterns across application categories

### 2. User Engagement Analysis
Focuses on behavioral intensity and segmentation:
- sessions per user
- total duration per user
- total traffic per user
- top users by usage
- outlier detection
- KMeans clustering for engagement segmentation
- application-level usage comparison

### 3. Experience Analysis
A placeholder module in the app for expanding into:
- service quality indicators
- latency and reliability metrics
- friction analysis
- user experience scoring

### 4. Satisfaction Analysis
A placeholder module for future work such as:
- NPS-style metrics
- churn signals
- retention storytelling
- satisfaction driver analysis

---

## 🏗️ Current Repository Structure

```text
Telecom_Analysis/
├── .vscode/
├── Data/
│   ├── Week1_challenge_data_source.xlsx
│   ├── cleaned_data.csv
│   ├── data.csv
│   ├── user_engagement.csv
│   └── user_experience_metrics.csv
├── Notebooks/
│   ├── .ipynb_checkpoints/
│   └── Analysis/
├── scripts/
│   ├── __pycache__/
│   ├── clean_telecom_data.py
│   ├── helper.py
│   └── plots.py
├── app.py
├── main.py
├── image.png
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
---

## 🛠️ Tech Stack
**Core Libraries**
- Python
- Pandas
- NumPy
- Scikit-learn
- Visualization
- Plotly
- Matplotlib
- Streamlit
---

**Workflow**
- Jupyter Notebooks
- Python scripts
- GitHub for version control
- uv / pip environment management

---
# ⚙️ Quick Start
- Clone the repository
- git clone https://github.com/Denis0242/Telecom_Analysis.git
- cd Telecom_Analysis
- Create and activate a virtual environment
- Using uv
- uv venv --python 3.11

**Windows:**
- .venv\Scripts\activate

**Mac/Linux**
- source .venv/bin/activate

**Then install dependencies:**

- pip install -r requirements.txt
**Run the dashboard**
- streamlit run app.py

---

# 📊 Dashboard Highlights
- The Streamlit app currently supports:
- loading the default dataset from Data/cleaned_data.csv
- optional upload of CSV or Excel telecom datasets
- dataset overview metrics
- missing value inspection
- handset analysis
- user behavior analysis
- application usage analysis
- engagement distributions
- outlier detection
- engagement clustering using KMeans

---
**A multi-page dashboard flow:**
- User Overview Analysis
- User Engagement Analysis
- Experience Analysis
- Satisfaction Analysis

---
# 📁 Available Data Assets
**The Data/ folder currently contains multiple assets that support different parts of the analysis:**
- cleaned_data.csv — main cleaned dataset used by the dashboard
- data.csv — original/base telecom dataset
- user_engagement.csv — engagement-focused derived data
- user_experience_metrics.csv — experience-focused derived data
- Week1_challenge_data_source.xlsx — source workbook version

_This is useful because it shows a workflow from raw/source data → cleaned data → targeted analysis outputs → dashboard delivery_.

--
# 🧠 Modeling and Analytics Approach
_This project currently demonstrates several important Product Data Science skills:_
- Exploratory Data Analys
- profiling user behavior
- identifying missing data
- comparing usage across app categories
- understanding device patterns and customer activity
- Feature Aggregation
- session counts
- total duration
- total traffic
- app-level data usage features
- Behavioral Segmentation
- scaling and normalization
- KMeans clustering
- elbow-method visualization
- cluster-level interpretation
- Dashboard Communication
- business-readable metrics
- visual summaries for stakeholder consumption
- interactive filtering and drill-down style exploration

# 💼 Why This Project Matters

This project is valuable for a Product Data Scientist portfolio because it shows the ability to:

translate raw telecom data into business-facing insights

connect behavioral metrics to customer segmentation

build interactive analytics products with Streamlit

combine EDA, ML, and dashboard storytelling in one repo

structure analysis work across scripts, datasets, notebooks, and app delivery

It is especially relevant to roles involving:

Product Analytics

Customer Analytics

Behavioral Analytics

Growth Analytics

Retention and Segmentation

Telecom / digital platform analytics

🔭 Suggested Future Enhancements

Planned or natural next improvements for this project include:

churn prediction modeling

retention cohort analysis

feature importance interpretation

satisfaction score modeling

anomaly detection for telecom usage spikes

richer experience metrics

business KPI layer

executive summary cards

deployment-ready Streamlit Cloud optimization

🖼️ Preview

The project includes a dashboard preview image:

![Telecom Analytics Dashboard](image.png)

You can keep this at the top so recruiters immediately see the visual output.

🤝 Author

**Denis Agyapong**

**Product Data Science / Data Analytics Portfolio Project**
