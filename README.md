---
runme:
  id: 01HTB4DNM8Z07BZSPET70MCD1E
  version: v3
---

# econ-667-contest-application

# Predictive Model for Effort Optimization in Machine Learning Competitions

## Overview

This project aims to develop a predictive model to optimize effort in machine learning competitions. The model leverages TensorFlow and includes hyperparameter tuning to enhance performance.

## Overarching question: 
"How do various contest design elements and participant behaviors quantitatively influence the outcomes of machine learning competitions, and what implications does this have for optimizing contest theory models to predict and enhance participant performance and innovation?"

This question is designed to encapsulate the various factors that affect participant performance in machine learning competitions, such as prize distribution, participant pool size, resource allocation, feedback mechanisms, problem complexity, prize allocation efficiency, and the use of collaboration tools. By answering this overarching question, you would be able to provide a comprehensive analysis of how these factors interact and influence each other, leading to a deeper understanding of contest dynamics in the context of machine learning competitions.
The original contribution of your research would be the development of an integrated model that not only predicts outcomes based on these factors but also offers practical insights for competition organizers to design contests that maximize participant engagement and elicit high-quality submissions. This model would also serve as a guide for participants to strategize their efforts and resource allocation to improve their chances of success.

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