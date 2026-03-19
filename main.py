from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from pricing_logic import surge_price,event_adjustment,weather_adjustment
from datetime import datetime
import random
from fastapi.middleware.cors import CORSMiddleware

# Auto weekend detection
def get_weekend():
    return 1 if datetime.now().weekday() >= 5 else 0

# Simulate available cars (based on location)
def get_available_cars(location):

    location = location.lower()

    if "airport" in location:
        return random.randint(10, 30)
    elif "downtown" in location:
        return random.randint(20, 50)
    else:
        return random.randint(40, 80)

# Simulate user rating
def get_user_rating():

    # Example: premium users more common
    return round(random.uniform(3.5, 5.0), 1)

app = FastAPI()

#Adding CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS
    allow_headers=["*"],
)

price_model = pickle.load(open("models/price_model.pkl","rb"))
demand_model = load_model("models/demand_lstm.keras", compile=False)


def get_past_demand():

    df = pd.read_csv("dataset/demand_data.csv")

    demand = df["bookings"].tail(5).values

    return demand


@app.get("/")
def home():

    return {"message":"Dynamic Pricing API Running"}


@app.post("/predict_price")
def predict_price(data: dict):

    location = data["location"]
    base_price = data["base_price"]

    # AUTO values
    available_cars = get_available_cars(location)
    weekend = get_weekend()
    rating = get_user_rating()
    weather = data["weather"]

    # Demand prediction
    past_demand = get_past_demand()
    past_demand = np.array(past_demand).reshape(1,5,1)

    predicted_demand = demand_model.predict(past_demand)[0][0]

    # Pricing logic
    price = surge_price(base_price, predicted_demand, available_cars)
    price = weather_adjustment(price, weather)

    # ML prediction
    features = [[predicted_demand, available_cars, weekend, rating, price]]
    final_price = price_model.predict(features)[0]

    return {
        "predicted_demand": float(predicted_demand),
        "dynamic_price": float(final_price),
        "available_cars": available_cars,
        "customer_rating": rating,
        "weekend": weekend
    }