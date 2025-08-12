import matplotlib.pyplot as plt


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


def plot_predicted_vs_actual(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # 45-degree reference line
    plt.xlabel('Actual Fantasy Points')
    plt.ylabel('Predicted Fantasy Points')
    plt.title(f'Predicted vs Actual Fantasy Points: {model_name}')
    plt.grid(True)
    plt.show()


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


def ask(question):
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")
