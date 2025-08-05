import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset and Preprocessing
dataset = pd.read_csv('archive/spreadspoke_scores.csv')

# Converting certain columns to numbers
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')

# Dependent variable is total score
dataset['y'] = (dataset['score_home'] + dataset['score_away'])

# Filtering the dataset to only relevant features
dataset = dataset[['over_under_line', 'y']]
dataset = dataset.dropna()
X = dataset[['over_under_line']].values
y = dataset.iloc[:, -1].values

# Splitting data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

plt.scatter(X_train, y_train, s=1)
plt.plot(X_train, regressor.predict(X_train))
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest Results: ")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
