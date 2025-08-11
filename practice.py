import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import data_processing
import features
import models
import visualization

# Data preprocessing
dataset = data_processing.load_and_preprocess_data('archive-2/yearly_player_stats_offense.csv',
                                                   'archive-2/yearly_team_stats_offense.csv')

position = data_processing.get_position()
data = data_processing.filter_dataset(dataset, position)

# Splitting features by position
if position == 'rb':
    features = features.get_features_by_position('rb')
elif position == 'wr':
    features = features.get_features_by_position('wr')
elif position == 'te':
    features = features.get_features_by_position('te')
else:
    raise ValueError("Position must be rb, wr, or te")

# Features and dependent variable
X = data[features].values
y = data['next_year_points'].values

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# BUILDING THE MODELS

# Building Linear Regression Model
results = models.train_and_evaluate_models(X_train, X_test, y_train, y_test)

lr = results['Linear Regression']
lr_model = lr[2]
lr_pred = lr[1]
lr_mse, lr_mae, lr_r2 = models.get_performance(y_test, lr_pred)

pr = results['Polynomial Regression']
pr_model = pr[2][1]
pr_pred = pr[1]
pr_mse, pr_mae, pr_r2 = models.get_performance(y_test, pr_pred)

dt = results['Decision Tree Regression']
dt_model = dt[2]
dt_pred = dt[1]
dt_mse, dt_mae, dt_r2 = models.get_performance(y_test, dt_pred)

rf = results['Random Forest Regression']
rf_model = rf[2]
rf_pred = rf[1]
rf_mse, rf_mae, rf_r2 = models.get_performance(y_test, rf_pred)

svr = results['SVR']
svr_model = svr[2][2]
svr_pred = svr[1]
svr_mse, svr_mae, svr_r2 = models.get_performance(y_test, svr_pred)

# Visualizing model metrics
model_names = ['Linear Regression', 'Polynomial Regression', 'Decision Tree Regression', 'Random Forest Regression',
               'SVR']
r2_scores = [lr_r2, pr_r2, dt_r2, rf_r2, svr_r2]
mse_scores = [lr_mse, pr_mse, dt_mse, rf_mse, svr_mse]
mae_scores = [lr_mae, pr_mae, dt_mae, rf_mae, svr_mae]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)
plt.ylabel('R² Score')
plt.title(f"R² Comparison of Regression Models for {position.upper()}'s")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(model_names, mse_scores)
plt.ylabel('MSE Score')
plt.title(f"MSE Comparison of Regression Models for {position.upper()}'s")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(model_names, mae_scores)
plt.ylabel('MAE Score')
plt.title(f"MAE Comparison of Regression Models for {position.upper()}'s")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Selecting the most accurate model based on R² values
optimal_model_index = r2_scores.index(max(r2_scores))
optimal_model = model_names[optimal_model_index]
print(f"\nOptimal Model: {optimal_model}; R²: {max(r2_scores)}")

# Default feature input
default_input = pd.DataFrame(X_train, columns=features).mean().to_dict()
