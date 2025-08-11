from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluate_results(results, position):
    model_names, r2_scores, mse_scores, mae_scores = [], [], [], []

    for name, (y_true, y_pred, _) in results.items():
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        model_names.append(name)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"\n---{name}---\nMSE: {mse}\nMAE: {mae}\nR²: {r2}")

    # Plot R²
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, r2_scores)
    plt.ylabel('R² Score')
    plt.title(f"R² Comparison of Regression Models for {position.upper()}")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.show()

    return model_names, r2_scores
