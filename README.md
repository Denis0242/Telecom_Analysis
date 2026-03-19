
# Telecom_Analysis 

![Telecom Analytics Dashboard](preview.png) 

# 📊 Telecom Customer Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-6f42c1.svg)](https://plotly.com/python/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-f7931e.svg)](https://scikit-learn.org/) 
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg)](https://pandas.pydata.org/)


A Product Data Science–oriented telecom analytics project that transforms telecom usage 
data into actionable insights across customer overview, engagement, experience, and
satisfaction workflows.

---

## 🚀 Project Summary

This project analyzes telecom customer behavior using Python, exploratory data analysis, feature
engineering, clustering, and an interactive Streamlit dashboard.

_It is designed to communicate the kind of workflow a **Product Data Scientist** would use to:_ 
- understand user behavior - profile customer engagement
- explore experience-related metrics - segment users into meaningful groups
- present insights through a clean interactive dashboard

- **The repository includes:**
- a **Streamlit app** for dashboard delivery
- **data files** for cleaned and intermediate analysis outputs
- **notebooks** for analysis work
- **Python scripts** for cleaning and helper utilities
  
  ---

## 🎯 Product Data Science Angle
**This project is positioned as a **product analytics + behavioral segmentation** case study in the telecom domain.**
_Key product questions this project helps answer:_ 
- Which users are most active and valuable?
- How does usage differ across apps like YouTube, Google, Netflix, Email, Gaming, and Social Media?
- Which users show high engagement versus low engagement?
- How can clustering help identify meaningful behavioral segments?
- How can telecom experience and satisfaction analysis be extended into retention and churn use cases?

--- 

## 🧩 Core Analysis Modules 
### 1. User Overview Analysis 
- dataset quality checks
- missing value analysis
- handset manufacturer and handset type exploration
- user/session distribution
- total usage patterns across application categories 

### 2. User Engagement Analysis 
- sessions per user
- total duration per user
- total traffic per user
- top users by usage
- outlier detection
- KMeans clustering for engagement segmentation
-  application-level usage comparison

---

### 3. Experience Analysis 
- service quality indicators
- latency and reliability metrics
- friction analysis
- user experience scoring
  
### 4. Satisfaction Analysis
- NPS-style metrics
- churn signals
- retention storytelling
- satisfaction driver analysis
  
---

## 🏗️ Project Structure 

```text
Telecom_Analysis/
├── Data/
├── Notebooks/
├── scripts/
├── app.py
├── requirements.txt
├── preview.png
└── README.md
```


## 🛠️ Tech Stack
_Core:_
- Python
- Pandas, NumPy
- Scikit-learn
  
_Visualization:_
- Plotly
- Matplotlib
- Streamlit

_Workflow:_
- Jupyter Notebooks
- GitHub
- Virtual Environments (uv / pip)

---

## ⚙️ Quick Start

**Clone the repository**
- git clone https://github.com/Denis0242/Telecom_Analysis.git
- cd Telecom_Analysis

**Create and activate a virtual environment**
- uv venv --python 3.11

Windows
- .venv\Scripts\activate

Mac/Linux
- source .venv/bin/activate

**Install dependencies**
- pip install -r requirements.txt

**Run the dashboard**
- streamlit run app.py

---

## 📊 Dashboard Highlights
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

## 📊 Multi-page Dashboard Flow
- User Overview Analysis
- User Engagement Analysis
- Experience Analysis
- Satisfaction Analysis

## 📁 Available Data Assets
- cleaned_data.csv — main cleaned dataset used by the dashboard
- data.csv — original telecom dataset
- user_engagement.csv — engagement-focused data
- user_experience_metrics.csv — experience-related features
- Week1_challenge_data_source.xlsx — source dataset

_This shows a full pipeline: raw data → cleaned data → analysis → dashboard_.

---

## 🧠 Modeling and Analytics Approach
- Exploratory Data Analysis
- profiling user behavior
- identifying missing data
- comparing usage across app categories
- Feature Aggregation
- session counts
- total duration
- total traffic
- app-level usage
- Behavioral Segmentation
- scaling and normalization
- KMeans clustering
- elbow method
- Dashboard Communication
- business-readable metrics
- visual storytelling
- interactive exploration

---

## 💼 Why This Project Matters
- translates raw telecom data into business insights
- applies product thinking to analytics
- builds interactive dashboards
- combines EDA + ML + storytelling
- demonstrates end-to-end workflow

_Relevant Roles:_
- Product Data Scientist
- Product Analyst
- Customer / Growth Analyst
- Behavioral Analytics

---

## 🔭 Suggested Future Enhancements
- churn prediction
- retention cohorts
- feature importance (SHAP)
- KPI layer (AARRR, North Star)
- A/B testing
- experience analytics
- Streamlit Cloud deployment

---

## 🤝 Author

**Denis Agyapong**

**Product Data Science / Data Analyst**
