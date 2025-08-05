import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset and Preprocessing
dataset = pd.read_csv('archive/spreadspoke_scores.csv')
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')
dataset['y'] = dataset['score_home'] + dataset['score_away']

# Filtering to only relevant features
dataset = dataset[['over_under_line', 'y']].dropna()
X = dataset[['over_under_line']].values
y = dataset['y'].values

# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (SVR requires it)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Training the SVR model
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)

# Predicting and Inverse Scaling
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Plotting
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_plot_scaled = sc_X.transform(X_plot)
y_plot_scaled = svr.predict(X_plot_scaled)
y_plot = sc_y.inverse_transform(y_plot_scaled.reshape(-1, 1))

plt.scatter(X_train, y_train, s=1)
plt.plot(X_plot, y_plot, color='red')
plt.title("SVR (RBF Kernel)")
plt.xlabel("Over/Under Line")
plt.ylabel("Total Score")
plt.show()

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Support Vector Regression Results: ")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
