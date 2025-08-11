from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    results['Linear Regression'] = (y_test, lr.predict(X_test), lr)

    # Polynomial Regression
    pr = PolynomialFeatures(degree=2)
    X_train_poly = pr.fit_transform(X_train)
    X_test_poly = pr.transform(X_test)
    lr2 = LinearRegression().fit(X_train_poly, y_train)
    results['Polynomial Regression'] = (y_test, lr2.predict(X_test_poly), (pr, lr2))

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X_train, y_train)
    results['Decision Tree Regression'] = (y_test, dt.predict(X_test), dt)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X_train, y_train)
    results['Random Forest Regression'] = (y_test, rf.predict(X_test), rf)

    # SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    svr = SVR(kernel='rbf').fit(X_train_scaled, y_train_scaled)
    svr_pred_scaled = svr.predict(X_test_scaled)
    svr_pred = scaler_y.inverse_transform(svr_pred_scaled.reshape(-1, 1)).ravel()
    results['SVR'] = (y_test, svr_pred, (scaler_X, scaler_y, svr))

    return results


def get_performance(y_test, pred):
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return mse, mae, r2


def predict_input(user_input, model_name, results, features, X_train):
    default_input = pd.DataFrame(X_train, columns=features).mean().to_dict()
    input_data = default_input.copy()

    # Update with user inputs to overwrite defaults
    input_data.update(user_input)

    _, _, model_obj = results[model_name]
    input_df = pd.DataFrame([input_data])[features]

    if model_name == 'Linear Regression':
        return model_obj.predict(input_df)[0]
    elif model_name == 'Polynomial Regression':
        pr, lr2 = model_obj
        return lr2.predict(pr.transform(input_df))[0]
    elif model_name == 'Decision Tree Regression':
        return model_obj.predict(input_df)[0]
    elif model_name == 'Random Forest Regression':
        return model_obj.predict(input_df)[0]
    elif model_name == 'SVR':
        scaler_X, scaler_y, svr = model_obj
        pred_scaled = svr.predict(scaler_X.transform(input_df))
        return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    else:
        raise ValueError("Invalid model name")


def optimal_model_r2(scores):
    model_names = ['Linear Regression', 'Polynomial Regression', 'Decision Tree Regression', 'Random Forest Regression',
                   'SVR']
    optimal_model_index = scores.index(max(scores))
    return model_names[optimal_model_index]
