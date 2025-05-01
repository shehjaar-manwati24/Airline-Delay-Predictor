import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the dataset
print("Step 1: Reading the dataset...")
df = pd.read_csv('Airline_Delay_Cause.csv')

# Step 2: Data cleaning and preprocessing
print("\nStep 2: Cleaning and preprocessing data...")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values in target variable
df = df.dropna(subset=['arr_delay'])

# Convert categorical variables to appropriate types
categorical_cols = ['carrier', 'carrier_name', 'airport', 'airport_name']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Step 3: Feature engineering and encoding
print("\nStep 3: Feature engineering and encoding...")

# Create delay ratio features
df['delay_ratio'] = df['arr_del15'] / df['arr_flights']
df['cancellation_ratio'] = df['arr_cancelled'] / df['arr_flights']
df['diversion_ratio'] = df['arr_diverted'] / df['arr_flights']

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['carrier', 'airport'])
df_encoded = df_encoded.drop(['carrier_name', 'airport_name'], axis=1)


# Select features for modeling
feature_cols = [
    'year', 'month', 'arr_flights', 'delay_ratio', 'cancellation_ratio', 
    'diversion_ratio', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 
    'late_aircraft_ct'
] + [col for col in df_encoded.columns if col.startswith(('carrier_', 'airport_'))]

X = df_encoded[feature_cols]
y = df_encoded['arr_delay']

# Step 4: Split into train-test sets
print("\nStep 4: Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
print("\nStep 5: Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
print("\nStep 6: Evaluating model performance...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} minutes")

# Step 7: Plot feature importance
print("\nStep 7: Plotting feature importance...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs Predicted Delay')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')

print("\nModel training and evaluation complete!")
print("Feature importance plot saved as 'feature_importance.png'")
print("Predicted vs actual plot saved as 'predicted_vs_actual.png'") 