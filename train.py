#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.base import clone
from pathlib import Path
import json
import joblib


# In[127]:


df = pd.read_csv('day.csv')

X = X = df.drop(columns=['cnt', 'dteday', 'casual', 'registered'])

y = df['cnt']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size = 0.5, random_state = 42
)

feature_cols = X_train.columns.tolist()


# In[129]:


imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()

X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(imputer.transform(X_val))
X_test = scaler.transform(imputer.transform(X_test))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rss(y_true, y_pred):
    y_true = np.asarray(y_true)
    return np.sum((y_true - y_pred) ** 2)

def evaluate_split(y, preds):
    return {
        "rss": rss(y, preds),
        "rmse": rmse(y, preds),
        "r2": r2_score(y, preds),
    }

# Candidate models
models = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=42),
    "lasso": Lasso(alpha=0.01, max_iter=10000, random_state=42),
}

# 1) Train on TRAIN, pick best on VAL by RMSE
val_results = {}
best_name = None
best_model = None
best_val_rmse = float("inf")

for name, m in models.items():
    model_candidate = clone(m)
    model_candidate.fit(X_train, y_train)

    val_preds = model_candidate.predict(X_val)
    metrics_val = evaluate_split(y_val, val_preds)

    val_results[name] = metrics_val

    if metrics_val["rmse"] < best_val_rmse:
        best_val_rmse = metrics_val["rmse"]
        best_name = name
        best_model = model_candidate

print("Validation results:")
for name, met in val_results.items():
    print(f"  {name:6s} | RMSE={met['rmse']:.2f} | R2={met['r2']:.3f} | RSS={met['rss']:.2f}")

print(f"\nBest model by VAL RMSE: {best_name} (RMSE={best_val_rmse:.2f})")

# 2) Refit best model on TRAIN+VAL
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([np.asarray(y_train), np.asarray(y_val)])

final_model = clone(models[best_name])
final_model.fit(X_trainval, y_trainval)

# 3) Final evaluation on TEST
test_preds = final_model.predict(X_test)
test_metrics = evaluate_split(y_test, test_preds)

print("\nTest metrics (final model):")
print(f"  RMSE={test_metrics['rmse']:.2f} | R2={test_metrics['r2']:.3f} | RSS={test_metrics['rss']:.2f}")


# In[133]:


# Create artifacts directory
Path("artifacts").mkdir(exist_ok=True)

# Save final refit model and preprocessors
joblib.dump(final_model, "artifacts/model.joblib")
joblib.dump(imputer, "artifacts/imputer.joblib")
joblib.dump(scaler, "artifacts/scaler.joblib")

# Save feature names (so predict.py uses the same ordering)
with open("artifacts/feature_names.json", "w") as f:
    json.dump(list(feature_cols), f)

print("\nSaved artifacts to ./artifacts/")


# In[ ]:




