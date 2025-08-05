import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive-2/yearly_player_stats_offense.csv')

# Filter by position
position = 'te'
dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()
dataset = dataset[dataset['position'] == position]

# Filtering by age
age = 30
dataset = dataset[dataset['age'] >= age]

dataset['player_name'] = dataset['player_name'].astype(str).str.strip().str.lower()
dataset['season'] = dataset['season'].astype(int)

# New dataset with only relevant data
data = dataset[
    [
        # Identifiers
        'player_name',
        'season',

        # Demographics
        'age',

        # Availability
        'games_missed',

        # Workload / Opportunity
        'touches',
        'rush_attempts',
        'targets',
        'career_touches',
        'offense_snaps',

        # Performance
        'fantasy_points_ppr',
        'total_yards',
        'total_tds',
    ]
]
# Sort the dataset to prepare for shifting
data = data.sort_values(by=['player_name', 'season'])

# Group by player and shift fantasy points to next year
data['next_year_points'] = data.groupby('player_name')['fantasy_points_ppr'].shift(-1)

# Drop rows where next year's data is missing
data = data.dropna()

# Features and dependent variable
X = data[[
    # Demographics
    'age',

    # Availability
    'games_missed',

    # Workload / Opportunity
    'touches',
    'rush_attempts',
    'targets',
    'career_touches',
    'offense_snaps',

    # Performance
    'fantasy_points_ppr',
    'total_yards',
    'total_tds',
]].values

y = data['next_year_points'].values

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

# Building Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, rf_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, rf_pred))
print("R² Score:", r2_score(y_test, rf_pred))
