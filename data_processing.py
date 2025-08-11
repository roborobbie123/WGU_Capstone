import pandas as pd


def load_and_preprocess_data(player_file, team_file):
    dataset = pd.read_csv(player_file)
    yearly_team_stats = pd.read_csv(team_file)

    yearly_team_stats = yearly_team_stats.rename(columns={
        'rush_attempts': 'total_rushes',
        'pass_attempts': 'team_passes'
    })
    dataset['position'] = dataset['position'].astype(str).str.strip().str.lower()

    added_columns = yearly_team_stats[['team', 'season', 'team_passes', 'total_rushes']]
    dataset = pd.merge(dataset, added_columns, on=['team', 'season'], how='left')
    return dataset


def filter_dataset(dataset, position):
    dataset_filtered = dataset.copy()
    dataset_filtered = dataset_filtered[dataset_filtered['position'] == position]
    dataset_filtered = dataset_filtered[(dataset_filtered['games_played_career'] > 0)]
    dataset_filtered['player_name'] = dataset_filtered['player_name'].astype(str).str.strip().str.lower()
    dataset_filtered['season'] = dataset_filtered['season'].astype(int)
    dataset_filtered['target_share'] = dataset_filtered['targets'] / dataset_filtered['team_passes']
    dataset_filtered['rush_share'] = dataset_filtered['rush_attempts'] / dataset_filtered['total_rushes']
    dataset_filtered['points'] = (
            dataset_filtered['rushing_yards'] * 0.1
            + dataset_filtered['receiving_yards'] * 0.1
            + dataset_filtered['receptions']
            + dataset_filtered['rush_touchdown'] * 6
            + dataset_filtered['receiving_touchdown'] * 6
    )

    data = dataset_filtered[
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
    ]

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

    return data


def get_position():
    positions = ['rb', 'wr', 'te']
    while True:
        position = input('Select a position (RB, WR, or TE): ').lower()
        if position in positions:
            return position
        else:
            print('Invalid position')
