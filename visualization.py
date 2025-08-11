from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


def position_avg(dataset):
    positions = ['wr', 'rb', 'te']
    data = dataset.copy()

    data['position'] = data['position'].str.lower().str.strip()

    filtered = data[data['position'].isin(positions)]

    avg_ppr_points = (
        filtered
        .groupby('position')['fantasy_points_ppr']
        .mean()
        .reindex(positions)
    )

    plt.figure(figsize=(8, 5))
    avg_ppr_points.index = avg_ppr_points.index.str.upper()  # make labels uppercase
    avg_ppr_points.plot(kind='bar', color=['orange', 'green', 'blue'])
    plt.title('Average PPR Points per Position')
    plt.xlabel('Position')
    plt.ylabel('Average PPR Points')
    plt.xticks(rotation=0)
    plt.show()


def evaluate_results(r2_scores, mse_scores, mae_scores, position):
    model_names = ['Linear Regression', 'Polynomial Regression', 'Decision Tree Regression', 'Random Forest Regression',
                   'SVR']

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, r2_scores)
    plt.ylabel('R² Score')
    plt.title(f"R² Comparison of Regression Models for {position.upper()}'s")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, mse_scores)
    plt.ylabel('MSE Score')
    plt.title(f"MSE Comparison of Regression Models for {position.upper()}'s")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, mae_scores)
    plt.ylabel('MAE Score')
    plt.title(f"MAE Comparison of Regression Models for {position.upper()}'s")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def get_default_input(X_train, features):

    return pd.DataFrame(X_train, columns=features).mean().to_dict()


def get_user_input(position):

    print('--- Enter player statistics from last season ---')
    while True:
        try:
            games_played = int(input('Games played: '))
            points = float(input('PPR points: '))
            targets = int(input('Targets: '))
            receiving_yards = float(input('Receiving yards: '))
            if position == 'rb':
                rushing_yards = float(input('Rushing yards: '))
            seasons = int(input('Total seasons played: '))
            break
        except ValueError:
            print('Invalid input, please enter numbers only.')

    user_input = {
        'games_played_season': games_played,
        'points': points,
        'targets': targets,
        'receiving_yards': receiving_yards,
        'seasons_played': seasons
    }
    if position == 'rb':
        user_input['rushing_yards'] = rushing_yards

    return user_input


def predict_input(user_input: dict, features, models_dict, scalers_dict, model_name='Random Forest Regression'):

    # Start from default input
    default_input = get_default_input(models_dict['X_train'], features)
    input_data = default_input.copy()

    # Update with user values
    input_data.update(user_input)

    # Make DataFrame in correct feature order
    input_df = pd.DataFrame([input_data])[features]

    if model_name == 'Linear Regression':
        return models_dict['lr'].predict(input_df)[0]
    elif model_name == 'Polynomial Regression':
        input_poly = models_dict['pr'].transform(input_df)
        return models_dict['lr2'].predict(input_poly)[0]
    elif model_name == 'Decision Tree Regression':
        return models_dict['dt'].predict(input_df)[0]
    elif model_name == 'Random Forest Regression':
        return models_dict['rf'].predict(input_df)[0]
    elif model_name == 'SVR':
        input_scaled = scalers_dict['scaler_X'].transform(input_df)
        pred_scaled = models_dict['svr'].predict(input_scaled)
        return scalers_dict['scaler_y'].inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    else:
        raise ValueError("Invalid model name")
