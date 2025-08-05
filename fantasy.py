import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive-2/yearly_player_stats_offense.csv')

# -------- DATA FILTERS -------------------------------------------------------

# Filter by position
position = 'te'
dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()
dataset = dataset[dataset['position'] == position]

# Filtering by age
# age = 23
# dataset = dataset[dataset['age'] >= age]

# Filtering by seasons played
# seasons_played = 0
# dataset = dataset[dataset['seasons_played'] >= seasons_played]

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
