import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset and Preprocessing
dataset = pd.read_csv('archive/spreadspoke_scores.csv')

# # Dropping all indoor games
# dataset['weather_detail'] = dataset['weather_detail'].astype(str).str.strip().str.lower()
# dataset = dataset[dataset['weather_detail'] != 'indoor']

# Converting certain columns to numbers
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['spread_favorite'] = pd.to_numeric(dataset['spread_favorite'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')
dataset['schedule_week'] = pd.to_numeric(dataset['schedule_week'], errors='coerce')

# Dependent variable is total score
dataset['y'] = (dataset['score_home'] + dataset['score_away'])

# Filtering the dataset to only relevant features
dataset = dataset[
    ['schedule_week', 'weather_temperature', 'weather_wind_mph',
     'weather_humidity',
     'spread_favorite', 'over_under_line', 'weather_detail', 'y']]
dataset = dataset.dropna()
X = dataset.iloc[:, 5].values
y = dataset.iloc[:, -1].values

line = dataset['over_under_line'].values

# # Label encoding binary features (playoff game)
# from sklearn.preprocessing import LabelEncoder
#
# le = LabelEncoder()
# X[:, 1] = le.fit_transform(X[:, 1])

# Encoding categorical data (stadium)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [6])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# Splitting data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train.reshape(-1, 1))
# X_test = sc.transform(X_test.reshape(-1, 1))

# BUILDING AND TESTING DIFFERENT MODELS

# Building the Linear Regression Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train)
y_pred_lr = lr.predict(X_test.reshape(-1, 1))

plt.scatter(X_train.reshape(-1, 1), y_train, s=1)
plt.plot(X_train.reshape(-1, 1), lr.predict(X_train.reshape(-1, 1)))
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print("Linear Regression Results: ")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Building the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X_train.reshape(-1, 1), y_train)
y_pred_rf = rf.predict(X_test.reshape(-1, 1))

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print("Mean Squared Error:", mse_rf)
print("Mean Absolute Error:", mae_rf)
print("R² Score:", r2_rf)






