import pandas as pd

# Load your yearly player data
df = pd.read_csv("archive-2/yearly_player_stats_offense.csv")

# Filter for Cooper Kupp
df = df[df['player_name'] == 'Cooper Kupp']

# Select relevant columns
df_selected = df[['season', 'receptions', 'receiving_yards', 'receiving_touchdown']]

# Print to verify
print(df_selected)

# Save to CSV
df_selected.to_csv('cooper_kupp_receiving_stats.csv', index=False)


