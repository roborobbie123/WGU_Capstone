import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive-2/yearly_player_stats_offense.csv')

# -------- DATA FILTERS -------------------------------------------------------

# Filter by position
position = 'wr'
dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()
dataset = dataset[dataset['position'] == position]

# Filtering by age
# age = 23
# dataset = dataset[dataset['age'] >= age]

# Filtering by seasons played
seasons_played = 5
dataset = dataset[dataset['seasons_played'] >= seasons_played]

# Filtering by targets
# targets = 20
# dataset = dataset[dataset['targets'] >= targets]

# Filtering by receiving yards
# yards = 1000
# dataset = dataset[dataset['receiving_yards'] >= yards]

# Filtering by rushing attempts
# attempts = 200
# dataset = dataset[dataset['rush_attempts'] >= attempts]

# Filter by fantasy points
# points = 200
# dataset = dataset[dataset['fantasy_points_ppr'] >= points]

# -------------------------------------------------------------------

dataset['player_name'] = dataset['player_name'].astype(str).str.strip().str.lower()
dataset['season'] = dataset['season'].astype(int)

# New dataset with only relevant data
data = dataset[
    [
        # Identifiers
        'player_name',
        'season',
        'draft_ovr',

        # Demographics
        'age',

        # Workload / Opportunity
        'rush_attempts',
        'targets',
        'receptions',
        'career_touches',
        'offense_snaps',
        'games_played_career',
        'games_played_season',

        # Performance
        'fantasy_points_ppr',
        'receiving_yards',
        'yards_after_catch',
        'rushing_yards',
        'total_tds',
        'career_fantasy_points_ppr'
    ]
].copy()

# Calculating ppg stats
data.loc[:, 'career_ppg'] = data['career_fantasy_points_ppr'] / data['games_played_career']
data.loc[:, 'season_ppg'] = data['fantasy_points_ppr'] / data['games_played_season']

# Sort the dataset to prepare for shifting
data = data.sort_values(by=['player_name', 'season'])

# Group by player and shift fantasy points to next year
data['next_year_points'] = data.groupby('player_name')['fantasy_points_ppr'].shift(-1)

# Drop rows where next year's data is missing
data = data.dropna()
data = data[(data['games_played_career'] > 0) & (data['games_played_season'] > 0)]

# Features and dependent variable
X = data[[
    # Demographics
    'age',
    # 'draft_ovr',

    # Workload / Opportunity
    'rush_attempts',
    'targets',
    'receptions',
    'career_touches',
    # 'offense_snaps',
    'games_played_career',
    'games_played_season',

    # Performance
    'fantasy_points_ppr',
    'receiving_yards',
    # 'yards_after_catch',
    'rushing_yards',
    'total_tds',
    # 'career_ppg',
    'season_ppg'
]].values

y = data['next_year_points'].values

# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# BUILDING THE MODELS

# Building Linear Regression Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Building Polynomial Regression Model

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2)
X_train_poly = pr.fit_transform(X_train)
X_test_poly = pr.transform(X_test)
lr2 = LinearRegression()
lr2.fit(X_train_poly, y_train)
pr_pred = lr2.predict(X_test_poly)

# Building Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Building SVR Model
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_trained_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.fit_transform(X_test)
y_trained_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_trained_scaled, y_trained_scaled)
svr_pred_scaled = svr.predict(X_test_scaled)
svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1))

# PERFORMANCE METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr_mse = mean_squared_error(y_test, y_pred)
lr_mae = mean_absolute_error(y_test, y_pred)
lr_r2 = r2_score(y_test, y_pred)
print("\n---Linear Regression---")
print("Mean Squared Error:", lr_mse)
print("Mean Absolute Error:", lr_mae)
print("R² Score:", lr_r2)

pr_mse = mean_squared_error(y_test, pr_pred)
pr_mae = mean_absolute_error(y_test, pr_pred)
pr_r2 = r2_score(y_test, pr_pred)
print("\n---Polynomial Regression---")
print("Mean Squared Error:", pr_mse)
print("Mean Absolute Error:", pr_mae)
print("R² Score:", pr_r2)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("\n---Random Forest---")
print("Mean Squared Error:", rf_mse)
print("Mean Absolute Error:", rf_mae)
print("R² Score:", rf_r2)

svr_mse = mean_squared_error(y_test, svr_pred)
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)
print("\n---SVR---")
print("Mean Squared Error:", svr_mse)
print("Mean Absolute Error:", svr_mae)
print("R² Score:", svr_r2)
