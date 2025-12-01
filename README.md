# 🏡 Immo Eliza – Real Estate Price Prediction (Machine Learning)

## 📊 Project Overview
![houseprediction](https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/wp-content/uploads/2021/12/housing_predictions-e1733474034542.jpg)
This project was developed for Immo Eliza to create a machine learning model for predicting property prices in Belgium. The solution implements and compares three different regression algorithms to find the most accurate price predictions.

## 🎯 Key Objectives

* Preprocess data for machine learning
* Apply regression in a real-life context
* Explore multiple machine learning models for regression
* Evaluate model performance using appropriate metrics
* Implement hyperparameter tuning and cross-validation to further improve models

## 📁 Repository Structure
```
IMMO-ELIZA-ML/
├── data/
│   └── immovlan_cleaned_file.csv
│
├── models/
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── medians.pkl
│   ├── modes.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── .ipynb
│   ├── immo-eliza-ml-final-version.ipynb
│   └── immo-eliza-ml-functions.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## 🗓 Timeline
--- Day 1 (24 Nov 2025) ---
- Introduction to the project
- Deeper understanding of regression
- Understanding of machine learning
- Design project structure 
- Create `notebooks/EDA.ipynb` and `notebooks/experiments.ipynb`
- Plan basic cleaning pipeline
- Create `src/data/load_data.py`, `src/data/main.py`, and `src/data/preprocess_data.py`

--- Day 2 (25 Nov 2025) ---
- Update files in the `tests` folder for unit-testing
- Unit-test the functions made the previous day
- Plan ML preprocessing pipeline and hyperparameters tuning
- Create `src/models/trainer.py` and `src/models/tuner.py`
- Test trainer functions in `notebooks/experiments.ipynb`

--- Day 3 (26 Nov 2025) ---
- Update files in the `tests` folder for unit-testing
- Unit-test the functions made the previous day
- Revisit EDA and models' trainer
- Improve `src/models/trainer.py`

--- Day 4 (27 Nov 2025 - FINAL DAY) ---
- Create `predict.py` (but not run)
- Skip hyperparameter tuning (was too heavy and time-consuming)
- Conduct cross validation for all models
- Run the final model trainers on `notebooks/experiments.ipynb`
- Save all the preprocessors and models
- Evaluate and compare models

## 🛠️ Technical Implementation
Models Implemented
1. Linear Regression - Baseline model
2. Random Forest - Ensemble method for non-linear relationships
3. XGBoost - Advanced gradient boosting for improved performance

## Data Pipeline
* Data Cleaning: Handling duplicates, missing values, and irrelevant columns
* Preprocessing: Imputation, one-hot encoding, and feature scaling
* Feature Selection: Correlation analysis and multicollinearity handling
* Model Training: Three different algorithms with proper train-test split
* Evaluation: Comprehensive metrics and overfitting analysis

## 📊 Performance Metrics
Models were evaluated using:
- R-squared (R²)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross-validation scores

## 🚀 Installation & Usage
Prerequisites
* Python 3.8+
* Virtual environment recommended

## Setup
'# Clone the repository
git clone https://github.com/yourusername/immo-eliza-ml.git

'# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

'# Install dependencies
pip install -r requirements.txt

## Running the Project
'# Train models
python train.py

'# Make predictions
python predict.py

## 📈 Results Summary
The project implemented three prediction models with the following key findings:
* XGBoost demonstrated the best overall performance
* Proper preprocessing significantly improved model accuracy
* Feature selection helped reduce overfitting

## 🔮 Future Improvements
* Hyperparameter optimization with GridSearchCV
* Explore more regularization techniques
* Feature engineering for additional predictive power
* Integration in pipeline.

# Package Usage
## Preprocessing pipeline (src/models/trainer.py)
The preprocessing pipeline includes imputation, data scaling (only for Ridge linear model), one-hot encoding, ... The pipelines will also be saved for each model in models/ as models/*_pipeline.pkl.

## Exploratory Data Analyses (notebooks/EDA.ipynb)
Used to visualise and inspect both raw and clean data.

## Training the models (notebooks/experiments.ipynb)
This notebook contains the step-by-step process of training the 3 models: Ridge, RandomForest, and XGBoost. It also has the interpretations of the results, both analytically (??) and in business terms. 

## 📌 Personal context note
This project was done as part of the AI & Data Science Bootcamp at BeCode (Ghent), class of 2025-2026. 
Feel free to reach out or connect with me on [LinkedIn](https://www.linkedin.com/in/aleksei-shashkov-612458308/)!