import joblib
import pandas as pd
import numpy as np
import time
import os

# Load the trained models
def load_models():
    try:
        soc_model = joblib.load('ml/models/soc_model.joblib')
        soh_model = joblib.load('ml/models/soh_model.joblib')
        print("Models loaded successfully.")
        return soc_model, soh_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def simulate_realtime_prediction(soc_model, soh_model):
    """
    Simulates receiving data from an ESP32 over Serial/Network and 
    making real-time predictions.
    """
    print("\nStarting Real-time Battery Monitoring Simulation...")
    print("-" * 50)
    print(f"{'Time (s)':<10} | {'Voltage (V)':<12} | {'Current (A)':<12} | {'Temp (C)':<10} | {'SOC (%)':<8} | {'SOH (%)':<8}")
    print("-" * 50)
    
    # Starting values for simulation
    current_time = 0
    simulated_soc = 98.0
    
    try:
        while True:
            # Simulate sensor readings (similar to what ESP32 would send)
            voltage = 3.6 + (simulated_soc / 100.0) * 0.5 + np.random.normal(0, 0.01)
            current = np.random.uniform(-5, 5)
            temperature = 28 + (abs(current) * 0.2) + np.random.normal(0, 0.5)
            internal_resistance = 52.0 + (100.0 - (simulated_soc * 0.98)) * 0.4
            
            # Prepare input for models
            input_data = pd.DataFrame([{
                'Voltage': voltage,
                'Current': current,
                'Temperature': temperature,
                'InternalResistance': internal_resistance
            }])
            
            # Make predictions
            predicted_soc = soc_model.predict(input_data)[0]
            predicted_soh = soh_model.predict(input_data)[0]
            
            # Display results
            print(f"{current_time:<10} | {voltage:<12.3f} | {current:<12.2f} | {temperature:<10.1f} | {predicted_soc:<8.1f} | {predicted_soh:<8.1f}")
            
            # Check for warnings
            if temperature > 45.0:
                print(">>> WARNING: CRITICAL BATTERY TEMPERATURE DETECTED! <<<")
            if predicted_soc < 15.0:
                print(">>> WARNING: LOW BATTERY! PLEASE CHARGE! <<<")
            if predicted_soh < 75.0:
                print(">>> CAUTION: BATTERY DEGRADATION DETECTED! <<<")

            # Update simulation state
            simulated_soc -= 0.5
            current_time += 1
            
            if simulated_soc < 0:
                print("\nBattery fully discharged. Simulation ended.")
                break
                
            time.sleep(1) # Delay between readings
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    soc_mod, soh_mod = load_models()
    if soc_mod and soh_mod:
        simulate_realtime_prediction(soc_mod, soh_mod)
