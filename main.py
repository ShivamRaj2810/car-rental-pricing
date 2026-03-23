from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from pricing_logic import surge_price, event_adjustment, weather_adjustment
from datetime import datetime
import random
from fastapi.middleware.cors import CORSMiddleware

import requests

# API_KEY="d4502c2f23a1ad875f373571329dbb0e"

# 🔐 ENV setup
from dotenv import load_dotenv
import os

load_dotenv()   # auto-detects .env in project root

API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("API KEY:", API_KEY)


# -------------------------------
# Utility Functions
# -------------------------------

# Weekend detection
def get_weekend():
    return 1 if datetime.now().weekday() >= 5 else 0


# Available cars simulation
def get_available_cars(location):
    location = location.lower()

    if "airport" in location:
        return random.randint(10, 30)
    elif "downtown" in location:
        return random.randint(20, 50)
    else:
        return random.randint(40, 80)


# User rating simulation
def get_user_rating():
    return round(random.uniform(3.5, 5.0), 1)


# 🌦 Weather from API (BACKEND)
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        print("Calling:", url)

        res = requests.get(url)
        data = res.json()

        print("FULL RESPONSE:", data)   # 👈 VERY IMPORTANT

        if data.get("cod") != 200:
            print("❌ API failed:", data)
            return "clear"

        w = data["weather"][0]["main"].lower()
        print("🌦 RAW WEATHER:", w)

        if "rain" in w or "drizzle" in w:
            return "rain"

        elif "storm" in w or "thunder" in w:
            return "storm"

        elif "cloud" in w:
            return "cloudy"

        elif "haze" in w or "mist" in w or "fog" in w or "smoke" in w:
            return "foggy"

        else:
            return "clear"

    except Exception as e:
        print("❌ ERROR:", e)
        return "clear"


# -------------------------------
# FastAPI App
# -------------------------------

app = FastAPI()

# CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load Models
# -------------------------------

price_model = pickle.load(open("models/price_model.pkl", "rb"))
demand_model = load_model("models/demand_lstm.keras", compile=False)


# -------------------------------
# Demand Data
# -------------------------------

def get_past_demand():
    df = pd.read_csv("dataset/demand_data.csv")
    demand = df["bookings"].tail(10).values
    return demand


# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def home():
    return {"message": "Dynamic Pricing API Running"}


@app.post("/predict_price")
def predict_price(data: dict):

    location = data["location"]
    base_price = data["base_price"]

    # AUTO values
    available_cars = get_available_cars(location)
    weekend = get_weekend()
    rating = get_user_rating()

    # 🌦 Fetch weather securely from backend
    weather = get_weather(location)

    # 📈 Demand prediction (LSTM)
    past_demand = get_past_demand()
    past_demand = np.array(past_demand).reshape(1, 10, 1)  # FIXED shape

    predicted_demand = demand_model.predict(past_demand)[0][0]

    # 💰 Pricing logic
    price = surge_price(base_price, predicted_demand, available_cars)
    price = weather_adjustment(price, weather)

    # 🤖 ML prediction
    features = [[predicted_demand, available_cars, weekend, rating, price]]
    final_price = price_model.predict(features)[0]

    weather = get_weather(location)
    print("FINAL WEATHER USED:", weather)

    # Response
    return {
        "predicted_demand": float(predicted_demand),
        "dynamic_price": float(final_price),
        "available_cars": available_cars,
        "customer_rating": rating,
        "weekend": weekend,
        "weather": weather
    }