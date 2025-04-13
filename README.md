# AttriNeuLift 🧠📈
Attribution Uplift Engine – A neural uplift modeling approach to estimate user conversion likelihood by attributing the contribution of features.

## 🔍 Overview
AttriNeuLift is a machine learning pipeline aimed at solving the uplift modeling problem — predicting the incremental impact of features (or "touchpoints") on a user's decision to convert.

It does so by learning from user data with 12 numerical features, targeting a binary classification output: conversion vs. no conversion.

Uplift modeling is particularly useful in:

- Marketing: Understanding which factors actually influence conversions.
- Causal Inference: Modeling the true effect of features rather than correlations.
- Personalization: Targeting users with interventions that are likely to shift their behavior.

This project uses neural networks for uplift estimation with implementations of S-learners and T-learners via the **causalml** library.

## 🧪 Key Components

- Criteo Dataset: Dataset containing anonymized user touchpoints.
- Feature Attribution: Understand which features drive uplift.
- CausalML: Uses XGBTRegressor from CausalML for uplift modeling.
- Neural Network: Used MLPTRegressor from CausalML
- Saved the model for future use using Pickle
- Evaluation Metrics:
  * Conversion lift and Uplift Gain
  * Contribution of each variable to the final purchase decision

## 🛠 Technologies Used
- Python 3.12+
- Jupyter Notebooks
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- causalml for uplift modeling
- PyTorch
- xgboost
- Pickle

## 📄 License
This project is licensed under the [MIT License](LICENSE).