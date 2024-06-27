# Import libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import datetime

# Step 1: Load data
df = pd.read_csv('health_data.csv')

# Step 2: Clean data
df = df.dropna()  # Drop missing vals

# Encode categorical vars
symps = [col for col in df.columns if col != 'disease']
df = pd.get_dummies(df, columns=symps)

# Step 3: Set features & target
X = df.drop('disease', axis=1)
y = df['disease']

# Split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
mdl = RandomForestClassifier(n_estimators=100, random_state=42)
mdl.fit(X_tr, y_tr)

# Step 5: Eval model
y_pred = mdl.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f'Accuracy: {acc}')

# Step 6: Save model
joblib.dump(mdl, 'health_mdl.pkl')

# Step 7: Log model
with open('mdl_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now()}: Accuracy: {acc}\n")
