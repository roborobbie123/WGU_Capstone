import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset and Preprocessing
dataset = pd.read_csv('archive/spreadspoke_scores.csv')
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['spread_favorite'] = pd.to_numeric(dataset['spread_favorite'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')

dataset['y'] = (dataset['score_home'] + dataset['score_away'])

dataset = dataset[['schedule_playoff', 'weather_temperature', 'weather_wind_mph', 'weather_humidity', 'spread_favorite', 'over_under_line', 'y']]
dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Analysing Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)



