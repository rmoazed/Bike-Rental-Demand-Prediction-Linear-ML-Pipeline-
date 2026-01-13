# Bike-Rental-Demand-Prediction (Linear-ML-Pipeline)
This project builds an end-to end machine learning pipeline to predict **daily bike rental demand** based on a variety of features included in the UCI Bike Rental dataset, including weather and calendar features. It focuses on **correct preprocessing, model selection, and reproducability**, with a simple deployment-ready prediction interface.
Problem: given daily weather and calendar information, predict the total number of bike rentals ('cnt') for that day. 

**Problem:**
This is a **regression problem** with real-world considerations such as:

  . proper train/validation/test splits
  
  . avoiding data leakage
  
  . consistent preprocessing
  
  . interpretable baseline models

**Data:**

  . **source:** UCI Machine Learning Repository - Bike Sharing Dataset
  
  . **file used:** 'day.csv'
  
  . **target:** 'cnt' (total daily rentals)

**Leakage Prevention**

The following columns were removed because they directly encocde the target:

  . 'casual'
  
  . 'registered'

  The raw date column 'dteday' was also removed after extracting calendar features.

**Approach**

**1.** **Data Splitting**

  . 70% train
  . 15% validation
  . 15% test

  (used fixed random seed for reproducibility)

**2. Preprocessing**

  . Missing values: **mean imputtion** (fit on training data only)
  . Feature scaling: **MinMaxScaler** (fit on training data only)
  . Same preprocessing applied to validation and test sets

**3. Models**

The following baseline models were trained and compared:

  . Linear Regression
  . Ridge Regression (L2 Regularization)
  . Lasso Regression (L1 Regularization)

**4. Model Selection**

  . Models are compared using **validation RMSE**
  . The best model is automatically selected
  . Final model is **refit on train + validation**
  . Final performance reported on the held-out test set

---

**Results**

Validation results:

  linear | RMSE=912.78 | R2=0.800 | RSS=91648431.91
  
  ridge  | RMSE=869.08 | R2=0.818 | RSS=83083174.42
  
  lasso  | RMSE=911.85 | R2=0.800 | RSS=91462357.46

Best model by VAL RMSE: ridge (RMSE=869.08)

Test metrics (final model):

  RMSE=813.11 | R2=0.826 | RSS=72726894.40

**Key Takeaways:**

  . Linear models explain ~80% of variance in daily demand
  . Performance is consistent across splits
  . Regularization does not significantly improve over the linear baseline for this dataset

**How to Run**

**1. Train models and save artifacts**

  '''bash

  python train.py

  This will:

  . Train all candidate models
  . Select the best one by validation RMSE
  . Refit it on train + validation sets
  . Save artifacts to artifacts/

**2. Run a Prediction**

  '''bash

  python predict.py --input example.json

  Example output: Predicted cnt: 1942.07

  Predictions are made using a JSON file with feature values:

  {
  
  "season": 2,
  
  "yr": 1,
  
  "mnth": 7,
  
  "holiday": 0,
  
  "weekday": 3,
  
  "workingday": 1,
  
  "weathersit": 1,
  
  "temp": 0.65,
  
  "atemp": 0.62,
  
  "hum": 0.55,
  
  "windspeed": 0.18
  
}

. Keys must match feature_names.json

. Missing values allowed (will be imputed)

. Extra keys are ignored

  
  

  
