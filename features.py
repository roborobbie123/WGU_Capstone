def get_features_by_position(position):
    features_dict = {
        'rb': ['age', 'seasons_played', 'career_touches', 'games_played_career',
               'games_played_season', 'touches', 'rush_attempts', 'rushing_yards',
               'targets', 'receptions', 'receiving_yards', 'rush_touchdown',
               'target_share', 'rush_share', 'offense_pct', 'season_ppg', 'points'],
        'wr': ['age', 'seasons_played', 'career_touches', 'games_played_career',
               'games_played_season', 'touches', 'targets', 'receptions',
               'receiving_yards', 'yards_after_catch', 'receiving_touchdown',
               'catch_pct', 'yards_per_reception', 'target_share', 'offense_snaps',
               'offense_pct', 'season_ppg', 'points'],
        'te': ['age', 'seasons_played', 'career_touches', 'games_played_career',
               'games_played_season', 'touches', 'targets', 'receptions',
               'receiving_yards', 'receiving_touchdown', 'yards_per_reception',
               'yards_after_catch', 'target_share', 'offense_pct', 'season_ppg', 'points']
    }
    return features_dict[position]