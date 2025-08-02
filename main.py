import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive/spreadspoke_scores.csv')

dataset['weather_wind_mph'] = pd.to_numeric(dataset['weather_wind_mph'], errors='coerce')
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')
dataset['y'] = ((dataset['score_home'] + dataset['score_away']) - dataset['over_under_line'])

dataset_clean = dataset.dropna(subset=['weather_wind_mph', 'y'])

X = dataset_clean['weather_wind_mph'].values.reshape(-1, 1)
y = dataset_clean['y'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


plt.scatter(X, y)
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('Wind (mph)')
plt.ylabel('Line - total')
plt.show()