import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive-2/yearly_player_stats_offense.csv')

# Dropping all QB stats
dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()
dataset = dataset[dataset['position'] != 'qb']

X = dataset[['touches', 'targets', 'offense_snaps']].values
y = dataset['fantasy_points_ppr'].values

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Building Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Performance Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

prediction = lr.predict([[500, 100, 700]])
print(prediction)