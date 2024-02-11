# Data Mining Project - Regression Analysis

## Overview
This project focuses on performing a detailed regression analysis of the "meta" dataset. The primary aim is to apply regression techniques to understand the characteristics of literature datasets processed by a learning algorithm.

## Phases of the Project

### 1. Dataset Analysis
- Load the dataset using scikit-learn's `fetch_openml`.
- Organize data into a pandas DataFrame and display the first few rows.

### 2. Preprocessing
- Remove nominal features and handle missing values.
- Create two datasets: D1 (missing values removed) and D2 (interpolated values).
- Apply standardization.

### 3. Regression
- Split datasets into training and test sets.
- Apply regression models (Linear Regression, Logistic Regression, etc.).
- Report results using MAE, MAPE, and SMAPE.
- Include appropriate graphs for analysis.
