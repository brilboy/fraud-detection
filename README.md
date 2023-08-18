# Credit Card Fraud Detection

## Description

This project aims to prevent credit card fraud and minimize financial losses by utilizing various machine learning techniques for fraud detection. The repository contains code and resources used to analyze and model credit card transaction data to accurately identify fraudulent transactions.

## Key Features

- Visualized Data: Employed `matplotlib` and `seaborn` to create insightful visualizations, allowing for a better understanding of data patterns and characteristics.
- Model Evaluation: Evaluated three different machine learning models - Isolation Forest, Gradient Boosting, and Neural Networks - to determine the most effective approach for fraud detection.
- User-Friendly API: Developed a user-facing API using `Flask` framework, enabling users to submit credit card transaction details and receive predictions regarding transaction authenticity.

## Project Overview

1. Data Exploration: Analyzed credit card transaction data to uncover patterns, anomalies, and potential features relevant to fraud detection.
2. Model Selection: Investigated the performance of Isolation Forest, Gradient Boosting, and Neural Networks to identify the optimal model for fraud detection.
3. API Development: Created an API with Flask, enabling users to interact with the fraud detection model via HTTP requests.
4. Deployment: Deployed the API on a server for real-time predictions and seamless integration into various applications.

## How to Use

1. Clone this repository: `git clone https://github.com/your-username/fraud-detection.git`
2. Install required dependencies: `pip install -r requirements.txt`
3. Explore the Jupyter notebooks in the `notebooks` directory to understand the data analysis and model evaluation process.
4. Navigate to the `app` directory and run `python app.py` to start the Flask API locally.
5. Access the API via `http://127.0.0.1:5000` and submit credit card transaction details to receive fraud predictions.

## Resources and References

- ChatGPT for providing assistance and guidance throughout the project.
- ["Machine Learning & AI Tutorial - Credit Card Fraud Detection"](https://www.youtube.com/watch?v=frM_7UMD_-A) by Krish Naik on YouTube for inspiration and insights.
- ["Credit Card Fraud Detection"](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to access dataset and further description of the case.