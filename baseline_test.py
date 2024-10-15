import pandas as pd
from sklearn.metrics import mean_squared_error
import os

# Load testing dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_directory, 'data', 'cs_kills_test.csv')

test_data = pd.read_csv(test_data_path, low_memory=False)

# Preprocessing: Drop columns that are not useful for analysis or have too many missing values
irrelevant_columns = ['date', 'raw_player_name', 'raw_team_name', 'raw_enemy_team_name', 'modified_ts']
test_data = test_data.drop(columns=irrelevant_columns)

# Handle mixed data types by converting non-numeric columns to numeric where possible
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# Handle missing values (for simplicity, fill with 0)
test_data = test_data.fillna(0)

# Define target
target = 'actual'
y_test = test_data[target]

# Baseline 1: Underdog line
underdog_line = test_data['line']  # Assuming 'line' is the underdog prediction
underdog_mse = mean_squared_error(y_test, underdog_line)
print(f"Mean Squared Error for Underdog Line: {underdog_mse}")

# Baseline 2: PrizePicks line
prizepicks_line = test_data['series_win_prob']  # Assuming 'series_win_prob' as PrizePicks prediction
prizepicks_mse = mean_squared_error(y_test, prizepicks_line)
print(f"Mean Squared Error for PrizePicks Line: {prizepicks_mse}")