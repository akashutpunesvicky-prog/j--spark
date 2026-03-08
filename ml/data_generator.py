import pandas as pd
import numpy as np
import os

def generate_battery_data(num_samples=1000):
    """
    Simulates battery data including voltage, current, temperature, 
    internal resistance, and cycles to predict SOC and SOH.
    """
    np.random.seed(42)
    
    # Time steps (simulated)
    time = np.arange(num_samples)
    
    # State of Charge (SOC) - Decreasing over time (simulating discharge)
    soc = np.linspace(100, 0, num_samples) + np.random.normal(0, 0.5, num_samples)
    soc = np.clip(soc, 0, 100)
    
    # State of Health (SOH) - Very slow degradation
    soh = 100 - (time / num_samples) * 5 + np.random.normal(0, 0.1, num_samples)
    soh = np.clip(soh, 0, 100)
    
    # Voltage (V) - Correlated with SOC (3.0V to 4.2V range)
    voltage = 3.0 + (soc / 100) * 1.2 + np.random.normal(0, 0.02, num_samples)
    
    # Current (A) - Random variations (positive for charge, negative for discharge)
    current = np.random.uniform(-10, 10, num_samples)
    
    # Temperature (C) - Rises with current and time
    temperature = 25 + (np.abs(current) * 0.5) + (time / num_samples) * 5 + np.random.normal(0, 1, num_samples)
    
    # Internal Resistance (mOhm) - Rises as SOH decreases
    internal_resistance = 50 + (100 - soh) * 2 + np.random.normal(0, 1, num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': time,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temperature,
        'InternalResistance': internal_resistance,
        'SOC': soc,
        'SOH': soh
    })
    
    return df

if __name__ == "__main__":
    print("Generating simulated battery data...")
    data = generate_battery_data(2000)
    
    # Create ml directory if not exists
    os.makedirs('ml', exist_ok=True)
    
    output_path = 'ml/battery_data.csv'
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(data.head())
