# DSM500 Final Project

## Bayesian Network Structure Learning and Explainability for E-commerce Conversion Prediction

This repository contains the Python implementation developed for my MSc Data Science and Artificial Intelligence final project. The study investigates the use of Bayesian Networks for predicting and explaining conversion behaviour from e-commerce clickstream sessions, using the RetailRocket recommender-system dataset.

The project evaluates whether a Bayesian Network can provide competitive predictive performance while offering an interpretable probabilistic representation of relationships between browsing behaviour and conversion. Logistic regression models are used as predictive baselines.

## Repository Structure

The analysis is organised into three notebooks that should be read in numerical order.

### `01_exploratory_data_analysis.ipynb`

Performs initial exploration of the RetailRocket event data, including:

- dataset structure and data-quality checks;
- event-type distributions;
- visitor and item activity;
- temporal browsing patterns;
- add-to-cart and transaction behaviour; and
- initial assessment of class imbalance.

The findings from this notebook inform the subsequent session construction and feature-engineering decisions.

### `02_preprocessing_feature_engineering.ipynb`

Transforms the raw event-level data into the session-level datasets used for modelling. This includes:

- removal of duplicate events;
- timestamp conversion and chronological ordering;
- construction of browsing sessions using a 30-minute inactivity threshold;
- session-level feature engineering;
- construction of the binary conversion target;
- prevention of target leakage;
- exploratory analysis of the resulting session-level features; and
- discretisation of variables for Bayesian Network modelling.

The notebook produces both continuous/session-level data for logistic regression and a discretised dataset for Bayesian Network development.

### `03_modelling_evaluation.ipynb`

Contains the main predictive modelling and Bayesian Network analysis, including:

- unweighted logistic regression baseline;
- class-weighted logistic regression baseline;
- evaluation using accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC;
- Bayesian Network structure learning using Hill Climb Search and BIC;
- sensitivity analysis using the K2 scoring criterion;
- Conditional Probability Table estimation;
- probabilistic inference for conversion;
- comparison of Bayesian Network and logistic regression performance; and
- decision-threshold sensitivity analysis.

## Dataset

The project uses the publicly available **RetailRocket recommender-system dataset**, containing anonymised e-commerce interaction events including product views, add-to-cart events and transactions.

The raw dataset is not included in this repository because of its size. The notebooks therefore assume that the original RetailRocket data files are available locally before execution.

## Modelling Approach

Individual clickstream events are first grouped into browsing sessions using a 30-minute inactivity threshold. Behavioural characteristics are then derived at session level, including event count, number of unique items viewed, add-to-cart behaviour, session duration and temporal characteristics.

Conversion is defined at session level according to whether a transaction occurred during the session. Transaction-derived information is excluded from the predictor set to prevent target leakage.

Two logistic regression specifications are used as predictive baselines. A discrete Bayesian Network is then learned from the training data using Hill Climb Search. BIC is used as the primary structure-learning score, with K2 used as a sensitivity analysis. Conditional Probability Tables are estimated from the training data and used for probabilistic inference.

## Evaluation

Because conversion is highly imbalanced, model performance is evaluated using multiple complementary metrics rather than accuracy alone:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices

Decision-threshold sensitivity is also examined to assess the precision-recall trade-off beyond the default classification threshold of 0.5.

## Software

The analysis was conducted in Python using libraries including:

- pandas
- NumPy
- scikit-learn
- pgmpy
- Matplotlib
- NetworkX

## Reproducibility

The notebooks are numbered according to the intended execution order. Intermediate datasets generated during preprocessing are used by the subsequent modelling notebook.

The repository is intended to accompany the final project report and provide sufficient implementation detail to demonstrate the data preparation, modelling, evaluation and probabilistic inference procedures used in the study.
