# IndustrialHR
🏭 Industrial Workforce Analytics using NLP & Machine Learning
📌 Project Overview

Understanding the industrial classification of the workforce is essential for economic planning and policy formulation in India. Traditional workforce classification data—covering main and marginal workers (excluding cultivators and agricultural laborers)—is often outdated and does not accurately reflect current employment trends.

This project applies data analytics, Natural Language Processing (NLP), and unsupervised machine learning to analyze and visualize India’s industrial workforce distribution across states and industries using an interactive Streamlit dashboard.

🎯 Problem Statement

In India, the classification of the workforce by industry, gender, and geography plays a critical role in understanding employment patterns and economic development. However, existing datasets are outdated and insufficient for modern decision-making. There is a need to clean, analyze, and reinterpret this data using advanced analytics techniques to generate actionable insights for policy makers and workforce planners.

💡 Proposed Solution

Clean and preprocess industrial workforce data for accuracy and consistency

Perform Exploratory Data Analysis (EDA) to identify key trends and patterns

Apply NLP techniques to analyze and group industry categories

Use unsupervised machine learning to understand structural similarities

Build interactive dashboards to visualize workforce distribution and insights

🔄 Project Workflow

Data → NLP → Machine Learning → Insights

Data Cleaning & Preparation

Remove missing and invalid industry records

Standardize state and industry names

Engineer key features such as total workforce

Exploratory Data Analysis (EDA)

Dataset preview and summary

Industry word cloud visualization

Distribution analysis across states and industries

Natural Language Processing (NLP)

Text preprocessing of industry categories

TF-IDF vectorization of industry descriptions

Unsupervised Machine Learning

KMeans clustering to analyze structural similarities between industries

Clustering used internally for analytical validation

Visualization & Insights

Interactive charts for state-wise and industry-wise analysis

Gender, rural–urban, and workforce share visualizations

📊 Dashboard Features
🏠 Home

Problem statement and project background

High-level solution approach

End-to-end workflow explanation

📈 EDA

Dataset preview

Industry Categories Word Cloud (unique values only)

🏛️ State-wise Analysis

Total workforce by state

Gender distribution by state

Urban vs rural workforce comparison

Urban workforce ratio trend

State-wise workforce share

🏭 Industry-wise Analysis

Total workforce by industry

Gender distribution by industry

Urban vs rural workforce by industry

Industry-wise workforce share

Female workforce participation trend

🛠️ Technologies Used

Python

Streamlit – interactive dashboard

Pandas & NumPy – data processing

Plotly – data visualization

Scikit-learn – NLP & machine learning

WordCloud – text visualization

📂 Project Structure
HR/
│
├── data/
│   └── processed/
│       └── clean_workers_data.csv
│
├── stremui.py
├── README.md
└── requirements.txt

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/industrial-workforce-analytics.git


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run stremui.py

📌 Key Takeaways

Workforce distribution varies significantly across states and industries

Urban employment dominates organized industrial sectors

Female participation remains comparatively lower across industries

NLP helps validate similarities among industry categories

The project demonstrates a complete classical ML workflow
