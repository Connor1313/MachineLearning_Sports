import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Reload the CSV file for 'cs_kills_train'
kills_train = pd.read_csv('cs_kills_train.csv', low_memory=False)



# Handling missing values: filling NaNs in 'role' and other necessary columns
kills_train.fillna({'role': 'Unknown'}, inplace=True)

# Encoding categorical features like 'team_name', 'raw_player_name', 'enemy_team_name'
label_encoder = LabelEncoder()

kills_train['team_name_encoded'] = label_encoder.fit_transform(kills_train['team_name'])
kills_train['raw_player_name_encoded'] = label_encoder.fit_transform(kills_train['raw_player_name'])
kills_train['enemy_team_name_encoded'] = label_encoder.fit_transform(kills_train['enemy_team_name'])
kills_train['role_encoded'] = label_encoder.fit_transform(kills_train['role'])

# Selecting features for the model
features = ['series_win_prob', 'team_name_encoded', 'raw_player_name_encoded', 'enemy_team_name_encoded', 'role_encoded', 'line']
target = 'actual'  # Actual number of kills

# Splitting the dataset into training and testing sets
X = kills_train[features]
y = kills_train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling NaN values by filling with the median
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Training the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display the Mean Squared Error and a few predictions vs actual values
mse, y_pred[:5], y_test[:5]

# Display the Mean Squared Error and a few predictions vs actual values
print(f"Mean Squared Error: {mse}")
print(f"Predictions: {y_pred[:5]}")
print(f"Actual values: {y_test[:5].values}")
