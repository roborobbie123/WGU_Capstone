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

# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# For smoother curve
sorted_indices = X_train[:, 0].argsort()
X_sorted = X_train[sorted_indices]
y_poly_sorted = lin_reg_2.predict(poly_reg.transform(X_sorted))

plt.scatter(X_train, y_train, s=1)
plt.plot(X_sorted, y_poly_sorted, color='red')
plt.title("Polynomial Regression Fit (Degree 4)")
plt.xlabel("Over/Under Line")
plt.ylabel("Total Score")
plt.show()

# Transform test data for polynomial model
X_test_poly = poly_reg.transform(X_test)
y_pred_poly = lin_reg_2.predict(X_test_poly)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression Results: ")
print("Mean Squared Error:", mse_poly)
print("Mean Absolute Error:", mae_poly)
print("R² Score:", r2_poly)
