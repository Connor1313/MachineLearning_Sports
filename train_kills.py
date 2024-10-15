import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load training and testing datasets
current_directory = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(current_directory, 'data', 'cs_kills_train.csv')
test_data_path = os.path.join(current_directory, 'data', 'cs_kills_test.csv')

train_data = pd.read_csv(train_data_path, low_memory=False)
test_data = pd.read_csv(test_data_path, low_memory=False)

# Print column names
print("Columns in train_data:")
print(train_data.columns.tolist())
print("\nColumns in test_data:")
print(test_data.columns.tolist())

# Preprocessing: Drop columns that are not useful for training or have too many missing values
irrelevant_columns = ['date', 'modified_ts']
train_data = train_data.drop(columns=irrelevant_columns, errors='ignore')
test_data = test_data.drop(columns=irrelevant_columns, errors='ignore')

# Keep the relevant columns as strings for evaluation purposes
test_data['source'] = test_data['source'].astype(str)
test_data['raw_player_name'] = test_data['raw_player_name'].astype(str)

# Ensure 'line' is numeric (if it's not already)
test_data['line'] = pd.to_numeric(test_data['line'], errors='coerce')

# Convert 'line_result' to numeric and handle errors
test_data['line_result'] = pd.to_numeric(test_data['line_result'], errors='coerce').fillna(0).astype(int)

# Feature engineering (only if 'kills' and 'rounds' columns exist)
if 'kills' in train_data.columns and 'rounds' in train_data.columns:
    train_data['kills_per_round'] = train_data['kills'] / train_data['rounds']
    test_data['kills_per_round'] = test_data['kills'] / test_data['rounds']

# Handle mixed data types by converting non-numeric columns to numeric where possible
for col in train_data.columns:
    if train_data[col].dtype == 'object' and col not in ['source', 'raw_player_name']:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        if col in test_data.columns and col not in ['source', 'raw_player_name']:
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# Handle missing values (for simplicity, fill with 0)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# Define relevant features
relevant_features = ['line', 'series_win_prob']

# Add any other relevant pre-match features that exist in your dataset
additional_features = ['map', 'opponent_rating', 'player_rating', 'player_recent_form']
for feature in additional_features:
    if feature in train_data.columns:
        relevant_features.append(feature)

# Define features and target
features = [col for col in relevant_features if col in train_data.columns]
target = 'actual'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Model Predictions vs Actual): {mse:.2f}")

# Calculate MSE for LINE vs actual kills
line_mse = mean_squared_error(test_data['actual'], test_data['line'])
print(f"Mean Squared Error (LINE vs Actual): {line_mse:.2f}")

# Calculate RMSE for both
model_rmse = np.sqrt(mse)
line_rmse = np.sqrt(line_mse)
print(f"Root Mean Squared Error (Model Predictions vs Actual): {model_rmse:.2f}")
print(f"Root Mean Squared Error (LINE vs Actual): {line_rmse:.2f}")

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Print predictions for 10 players
sample_players = test_data.head(10)
sample_predictions = y_pred[:10]

for i, player in sample_players.iterrows():
    source = player['source']
    line_result = player['line_result']
    line = player['line']
    result_str = "Win" if line_result == 1 else "Loss" if line_result == -1 else "Push"
    
    print(f"Player: {player['raw_player_name']}, Predicted Kills: {sample_predictions[i]:.2f}, Source: {source}, LINE: {line:.1f}, Line Result: {result_str}, Actual Kills: {player['actual']:.2f}")
