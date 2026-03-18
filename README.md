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


