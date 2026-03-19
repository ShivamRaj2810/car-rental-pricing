import pandas as pd
import random
import time
from datetime import datetime

file_path = "dataset/demand_data.csv"
MAX_ROWS = 10000

while True:

    df = pd.read_csv(file_path)

    hour = datetime.now().hour

    demand = random.randint(20,40)

    if 8 <= hour <= 10 or 17 <= hour <= 20:
        demand += random.randint(40,70)

    if datetime.now().weekday() >= 5:
        demand += random.randint(20,40)

    new_row = {
        "timestamp": datetime.now(),
        "bookings": demand
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # keep only last 500 rows
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)

    df.to_csv(file_path, index=False)

    print("New demand:", demand)

    time.sleep(10)