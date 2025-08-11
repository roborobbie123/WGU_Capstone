from sklearn.model_selection import train_test_split
import data_processing
import features
import models
import visualization

# DATA PREPROCESSING
dataset = data_processing.load_and_preprocess_data('archive-2/yearly_player_stats_offense.csv',
                                                   'archive-2/yearly_team_stats_offense.csv')

position = data_processing.get_position()
data = data_processing.filter_dataset(dataset, position)

# Getting features by position
if position == 'rb':
    features = features.get_features_by_position('rb')
elif position == 'wr':
    features = features.get_features_by_position('wr')
elif position == 'te':
    features = features.get_features_by_position('te')
else:
    raise ValueError("Position must be rb, wr, or te")

# FEATURES AND DEPENDENT VARIABLE
X = data[features].values
y = data['next_year_points'].values

# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# BUILDING THE MODELS
results = models.train_and_evaluate_models(X_train, X_test, y_train, y_test)

lr = results['Linear Regression']
lr_model = lr[2]
lr_pred = lr[1]
lr_mse, lr_mae, lr_r2 = models.get_performance(y_test, lr_pred)

pr = results['Polynomial Regression']
pr_model = pr[2][1]
pr_pred = pr[1]
pr_mse, pr_mae, pr_r2 = models.get_performance(y_test, pr_pred)

dt = results['Decision Tree Regression']
dt_model = dt[2]
dt_pred = dt[1]
dt_mse, dt_mae, dt_r2 = models.get_performance(y_test, dt_pred)

rf = results['Random Forest Regression']
rf_model = rf[2]
rf_pred = rf[1]
rf_mse, rf_mae, rf_r2 = models.get_performance(y_test, rf_pred)

svr = results['SVR']
svr_model = svr[2][2]
svr_pred = svr[1]
svr_mse, svr_mae, svr_r2 = models.get_performance(y_test, svr_pred)

# VISUALIZING POSITION AVERAGE AND MODEL METRICS
models_list = [lr_model, pr_model, dt_model, rf_model, svr_model]
r2_scores = [lr_r2, pr_r2, dt_r2, rf_r2, svr_r2]
mse_scores = [lr_mse, pr_mse, dt_mse, rf_mse, svr_mse]
mae_scores = [lr_mae, pr_mae, dt_mae, rf_mae, svr_mae]

visualization.position_avg(dataset)

visualization.evaluate_results(r2_scores, mse_scores, mae_scores, position)

# Selecting the most accurate model based on R² values
optimal_model = models.optimal_model_r2(r2_scores)
print(f"Optimal Model based on R2: {optimal_model}")

user_input = visualization.get_user_input(position)
prediction = models.predict_input(user_input, optimal_model, results, features, X_train)
print(f"Expected PPR points: {prediction}")
