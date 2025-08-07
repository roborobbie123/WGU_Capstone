import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data preprocessing
dataset = pd.read_csv('archive-2/yearly_player_stats_offense.csv')
yearly_team_stats = pd.read_csv('archive-2/yearly_team_stats_offense.csv')
yearly_team_stats = yearly_team_stats.rename(columns={'rush_attempts': 'total_rushes'})
yearly_team_stats = yearly_team_stats.rename(columns={'pass_attempts': 'team_passes'})

# Adding team total passes and rushes to data
added_columns = yearly_team_stats[['team', 'season', 'team_passes', 'total_rushes']]
dataset = pd.merge(dataset, added_columns, on=['team', 'season'], how='left')

# -------- DATA FILTERS -------------------------------------------------------
positions = ['rb', 'wr', 'te']
while True:
    position = input('Select a position (RB, WR, or TE): ').lower()
    if position in positions:
        break
    else:
        print('Invalid position')

# Filter by position
dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()
dataset = dataset[dataset['position'] == position]

# Excludes rookies
dataset = dataset[(dataset['games_played_career'] > 0)]
# -------------------------------------------------------------------

dataset['player_name'] = dataset['player_name'].astype(str).str.strip().str.lower()
dataset['season'] = dataset['season'].astype(int)
dataset['target_share'] = dataset['targets'] / dataset['team_passes']
dataset['rush_share'] = dataset['rush_attempts'] / dataset['total_rushes']
dataset['points'] = (dataset['rushing_yards'] * 0.1) + (dataset['receiving_yards'] * 0.1) + dataset['receptions'] + (
        dataset['rush_touchdown'] * 6) + (dataset['receiving_touchdown'] * 6)

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
        'target_share',
        'rush_share',
        'career_touches',
        'offense_pct',
        'offense_snaps',
        'games_played_career',
        'games_played_season',
        'touches',
        'seasons_played',

        # Efficiency
        'tackled_for_loss',
        'first_down_rush',
        'third_down_converted',
        'third_down_failed',
        'season_average_ypc',

        # Performance
        'points',
        'receiving_yards',
        'yards_after_catch',
        'rushing_yards',
        'rush_touchdown',
        'receiving_touchdown'
    ]
].copy()

# Calculating ppg stats
data.loc[:, 'season_ppg'] = data['points'] / data['games_played_season']
data['catch_pct'] = data['receptions'] / data['targets']
data['yards_per_reception'] = data['receiving_yards'] / data['receptions']

# Sort the dataset to prepare for shifting
data = data.sort_values(by=['player_name', 'season'])

# Group by player and shift fantasy points to next year
data['next_year_points'] = data.groupby('player_name')['points'].shift(-1)

# Drop rows where next year's data is missing
data = data.dropna()
data = data[(data['games_played_career'] > 0) & (data['games_played_season'] > 0)]

# Splitting features by position
rb_features = [
    'age',
    'seasons_played',
    'career_touches',
    'games_played_career',
    'games_played_season',
    'touches',
    'rush_attempts',
    'rushing_yards',
    'targets',
    'receptions',
    'receiving_yards',
    'rush_touchdown',
    'target_share',
    'rush_share',
    'offense_pct',
    'season_ppg',
    'points',
]
wr_features = [
    'age',
    'seasons_played',
    'career_touches',
    'games_played_career',
    'games_played_season',
    'touches',
    'targets',
    'receptions',
    'receiving_yards',
    'yards_after_catch',
    'receiving_touchdown',
    'catch_pct',
    'yards_per_reception',
    'target_share',
    'offense_snaps',
    'offense_pct',
    'season_ppg',
    'points',

]
te_features = [
    'age',
    'seasons_played',
    'career_touches',
    'games_played_career',
    'games_played_season',
    'touches',
    'targets',
    'receptions',
    'receiving_yards',
    'receiving_touchdown',
    'yards_per_reception',
    'yards_after_catch',
    'target_share',
    'offense_pct',
    'season_ppg',
    'points',
]

if position == 'rb':
    features = rb_features
elif position == 'wr':
    features = wr_features
elif position == 'te':
    features = te_features
else:
    raise ValueError("Position must be rb, wr, or te")

# Features and dependent variable
X = data[features].values
y = data['next_year_points'].values

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# BUILDING THE MODELS

# Building Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Building Polynomial Regression Model
pr = PolynomialFeatures(degree=2)
X_train_poly = pr.fit_transform(X_train)
X_test_poly = pr.transform(X_test)
lr2 = LinearRegression()
lr2.fit(X_train_poly, y_train)
pr_pred = lr2.predict(X_test_poly)

# Building Decision Tree Model
dt = DecisionTreeRegressor(max_depth=5, random_state=0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Building Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Building SVR Model
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_trained_scaled = scaler_X.fit_transform(pd.DataFrame(X_train, columns=features))
X_test_scaled = scaler_X.transform(pd.DataFrame(X_test, columns=features))
y_trained_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf')
svr.fit(X_trained_scaled, y_trained_scaled)
svr_pred_scaled = svr.predict(X_test_scaled)
svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1))

# PERFORMANCE METRICS

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

dt_mse = mean_squared_error(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)
print("\n---Decision Tree Regression---")
print("Mean Squared Error:", dt_mse)
print("Mean Absolute Error:", dt_mae)
print("R² Score:", dt_r2)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("\n---Random Forest Regression---")
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

# Visualizing model metrics
model_names = ['Linear Regression', 'Polynomial Regression', 'Decision Tree Regression', 'Random Forest Regression',
               'SVR']
r2_scores = [lr_r2, pr_r2, dt_r2, rf_r2, svr_r2]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)
plt.ylabel('R² Score')
plt.title(f"R² Comparison of Regression Models for {position.upper()}'s")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Selecting the most accurate model based on R² values
optimal_model_index = r2_scores.index(max(r2_scores))
optimal_model = model_names[optimal_model_index]
print(f"\nOptimal Model: {optimal_model}; R²: {max(r2_scores)}")

# Default feature input
default_input = pd.DataFrame(X_train, columns=features).mean().to_dict()


# Making prediction based on optimal model
def predict_from_partial_input(user_input: dict, model_name='Random Forest Regression'):
    # Start from default
    input_data = default_input.copy()

    # Update with user-provided values
    input_data.update(user_input)

    # Convert to DataFrame for consistency
    input_df = pd.DataFrame([input_data])[features]

    # Apply model-specific transformation
    if model_name == 'Linear Regression':
        return lr.predict(input_df)[0]
    elif model_name == 'Polynomial Regression':
        input_poly = pr.transform(input_df)
        return lr2.predict(input_poly)[0]
    elif model_name == 'Decision Tree Regression':
        return dt.predict(input_df)[0]
    elif model_name == 'Random Forest Regression':
        return rf.predict(input_df)[0]
    elif model_name == 'SVR':
        input_scaled = scaler_X.transform(input_df)
        pred_scaled = svr.predict(input_scaled)
        return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    else:
        raise ValueError("Invalid model name")


print('--- Player statistics ---')
while True:
    try:
        games_played = int(input('Games played: '))
        points = int(input('PPR points: '))
        targets = int(input('Targets: '))
        receiving_yards = int(input('Receiving yards: '))
        if position == 'rb':
            rushing_yards = int(input('Rushing yards: '))
        seasons = int(input('Total seasons played: '))
        break
    except ValueError:
        print('Invalid input')

user_input = {
    'games_played_season': games_played,
    'points': points,
    'targets': targets,
    'receiving_yards': receiving_yards,
    'seasons_played': seasons
}
if position == 'rb':
    user_input['rushing_yards'] = rushing_yards

output = predict_from_partial_input(user_input, optimal_model)
print(f"Predicted PPR Points: {output}")
