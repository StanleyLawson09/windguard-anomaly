# Simulation script for a continous stream of Iot sensor data
import time
import random
import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
DELAY_SECONDS = 1.5  # Delay between requests to simulate a continuous stream of data in real life scenarios

# Dataset loading
df = pd.read_csv("data/ai4i2020.csv")

# Dataset columns mapping to the API input fields
column_mapping = {
    "air_temperature": "air_temperature",
    "process_temperature": "process_temperature",
    "rotational_speed": "rotational_speed",
    "torque": "torque",
    "tool_wear": "tool_wear"
}

print(f"Starting stream simulation - sending {len(df)} readings to the API at {API_URL}")
print("-" * 65)

for index, row in df.iterrows():
    # Prepare the payload
    payload = {"type": str(row.get("type", "L"))}
    for dataset_col, api_field in column_mapping.items():
        payload[api_field] = round(float(row[dataset_col]), 2)

    # Send the request to the API
    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        result = response.json()

        # Check if response has errors
        if response.status_code != 200:
            print(f"[ERROR] API returned status {response.status_code}: {result}")
            break

        actual = int(row.get("machine_failure", 0))
        predicted = result.get("label", "Unknown")
        prob = result.get("failure_probability", 0)

        status = "🚨 FAILURE" if predicted == "Failure" else "  normal"
        marker = " ACTUAL FAILURE" if actual == 1 else ""

        print(f"[{index:5d}] {status} prob={prob:.3f} "
              f"temp={payload['air_temperature']:.1f}K "            
              f"torque={payload['torque']:.1f}Nm{marker} ")
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection error. Is the API(uvicorn) running?")
        break
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        break

    time.sleep(DELAY_SECONDS + random.uniform(-0.3, 0.3))  # Add some randomness to the delay to simulate real-world conditions
