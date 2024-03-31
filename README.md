---
runme:
  id: 01HTB4DNM8Z07BZSPET70MCD1E
  version: v3
---

# econ-667-contest-application

# Predictive Model for Effort Optimization in Machine Learning Competitions

## Overview
This project aims to develop a predictive model to optimize effort in machine learning competitions. The model leverages TensorFlow and includes hyperparameter tuning to enhance performance.

## Installation Instructions
1. Clone the repository:
`git clone https://github.com/devyn-miller/econ-667-contest-application.git`
2. Install required packages:
pip install -r requirements.txt


## Usage
- Run data collection: `python data_collection/collect_data.py`
- Preprocess data: `python data_preprocessing/preprocess_data.py`
- Perform EDA: Open and run `exploratory_data_analysis/eda.ipynb`
- Train the model: `python model/train_model.py`
- Tune hyperparameters: `python model/tune_hyperparameters.py`
- Evaluate the model: `python evaluation/evaluate_model.py`
- Run the app: `streamlit run deployment/app.py`

## Contributors
- Devyn Miller

## License
This project is licensed under the MIT License - see the LICENSE file for details.