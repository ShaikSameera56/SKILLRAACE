# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime

# Step 1: Load data
df = pd.read_csv('ride_requests.csv')

# Step 2: Preprocess data
df = df.dropna()  # Remove missing values

# Feature engineering
df['hr'] = pd.to_datetime(df['timestamp']).dt.hour
df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['wknd'] = df['dow'] >= 5

# Step 3: EDA
sns.lineplot(data=df, x='hr', y='ride_requests')
plt.show()

# Step 4: Select features
feat = ['hr', 'dow', 'wknd', 'temperature', 'weather_condition']
X = df[feat]
y = df['ride_requests']

# Split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
mdl = RandomForestRegressor(n_estimators=100, random_state=42)
mdl.fit(X_tr, y_tr)

# Step 6: Evaluate model
y_pred = mdl.predict(X_te)
mae = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Step 7: Save model
joblib.dump(mdl, 'ride_mdl.pkl')

# Step 8: Monitor model
# Simple logging
with open('model_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now()}: MAE: {mae}, RMSE: {rmse}\n")
