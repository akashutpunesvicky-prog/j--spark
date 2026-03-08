import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_and_save_models(data_path='ml/battery_data.csv'):
    # Load data
    df = pd.read_csv(data_path)
    
    # Features and Targets
    # Features: Voltage, Current, Temperature, InternalResistance
    X = df[['Voltage', 'Current', 'Temperature', 'InternalResistance']]
    y_soc = df['SOC']
    y_soh = df['SOH']
    
    # Split data
    X_train, X_test, y_soc_train, y_soc_test, y_soh_train, y_soh_test = train_test_split(
        X, y_soc, y_soh, test_size=0.2, random_state=42
    )
    
    # Train SOC Model
    print("Training SOC model...")
    soc_model = RandomForestRegressor(n_estimators=100, random_state=42)
    soc_model.fit(X_train, y_soc_train)
    
    # Train SOH Model
    print("Training SOH model...")
    soh_model = RandomForestRegressor(n_estimators=100, random_state=42)
    soh_model.fit(X_train, y_soh_train)
    
    # Evaluate SOC Model
    soc_preds = soc_model.predict(X_test)
    soc_mae = mean_absolute_error(y_soc_test, soc_preds)
    soc_r2 = r2_score(y_soc_test, soc_preds)
    print(f"SOC Model - MAE: {soc_mae:.4f}, R2: {soc_r2:.4f}")
    
    # Evaluate SOH Model
    soh_preds = soh_model.predict(X_test)
    soh_mae = mean_absolute_error(y_soh_test, soh_preds)
    soh_r2 = r2_score(y_soh_test, soh_preds)
    print(f"SOH Model - MAE: {soh_mae:.4f}, R2: {soh_r2:.4f}")
    
    # Save models
    os.makedirs('ml/models', exist_ok=True)
    joblib.dump(soc_model, 'ml/models/soc_model.joblib')
    joblib.dump(soh_model, 'ml/models/soh_model.joblib')
    print("Models saved to ml/models/")
    
    # Simple prediction test
    sample = X_test.iloc[0:1]
    print(f"\nSample Prediction Test:")
    print(f"Input: {sample.to_dict(orient='records')[0]}")
    print(f"Predicted SOC: {soc_model.predict(sample)[0]:.2f}% (Actual: {y_soc_test.iloc[0]:.2f}%)")
    print(f"Predicted SOH: {soh_model.predict(sample)[0]:.2f}% (Actual: {y_soh_test.iloc[0]:.2f}%)")

if __name__ == "__main__":
    train_and_save_models()
