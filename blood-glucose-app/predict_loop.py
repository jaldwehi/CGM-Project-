import time
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/api/v1/predict"

while True:
    try:
        r = requests.get(API_URL)
        data = r.json()

        if "prediction" not in data:
            print("Waiting for enough data...", data)
            time.sleep(5)
            continue

        print(
            datetime.now().strftime("%H:%M:%S"),
            "Predicted glucose:",
            float(data["prediction"])
        )

    except Exception as e:
        print("Error:", e)

    time.sleep(60* 60)   # 1 hour 
