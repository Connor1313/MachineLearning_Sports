import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load training and testing datasets
train_data = pd.read_csv('/data/cs_kills_train.csv')
test_data = pd.read_csv('/data/cs_kills_test.csv')

# Preprocessing: Drop columns that are not useful for training or have too many missing values
irrelevant_columns = ['date', 'raw_player_name', 'raw_team_name', 'raw_enemy_team_name', 'modified_ts']
train_data = train_data.drop(columns=irrelevant_columns)
test_data = test_data.drop(columns=irrelevant_columns)

# Handle missing values (for simplicity, fill with 0)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

# Define features and target
features = [col for col in train_data.columns if col not in ['actual']]
target = 'actual'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Split training data for validation (optional, can skip if using the whole training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model (using Random Forest Regressor as an example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Optionally, evaluate on the validation set as well
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error on Validation Set: {val_mse}")