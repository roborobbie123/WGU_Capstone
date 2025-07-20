import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('archive/spreadspoke_scores.csv')
X = dataset[['schedule_playoff', 'over_under_line']]
dataset['over_under_line'] = pd.to_numeric(dataset['over_under_line'], errors='coerce')
dataset['score_home'] = pd.to_numeric(dataset['score_home'], errors='coerce')
dataset['score_away'] = pd.to_numeric(dataset['score_away'], errors='coerce')
y = (dataset['score_home'] + dataset['score_away']) > dataset['over_under_line']

print(X)
print(y)