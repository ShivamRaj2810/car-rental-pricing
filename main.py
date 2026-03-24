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

import sqlite3
from passlib.hash import bcrypt

from pydantic import BaseModel,EmailStr



# 🔐 ENV setup
from dotenv import load_dotenv
import os

load_dotenv()   # auto-detects .env in project root

API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("API KEY:", API_KEY)


# -------------------------------
# User Database Setup
# -------------------------------

def create_user_table():
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()

create_user_table()


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


#REGISTER

# Request body model
class User(BaseModel):
    email: EmailStr
    password: str

@app.post("/register")
def register(user: User):
    email = user.email
    password = user.password

    if not email or not password:
        return {"error": "Email and password required"}

    hashed_password = bcrypt.hash(password)

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hashed_password)
        )
        conn.commit()
        return {"message": "User registered successfully ✅"}

    except sqlite3.IntegrityError:
        return {"error": "email already exists ❌"}

    finally:
        conn.close()

#LOGIN
@app.post("/login")
def login(data: dict):
    email = data.get("email")
    password = data.get("password")

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    conn.close()

    if user and bcrypt.verify(password, user[2]):
        return {"message": "Login successful ✅"}
    else:
        return {"error": "Invalid credentials ❌"}


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
    
    import pandas as pd

    input_data = pd.DataFrame([{
        "predicted_demand": predicted_demand,
        "available_cars": available_cars,
        "weekend": weekend,
        "customer_rating": rating,
        "base_price": price
    }])

    final_price = price_model.predict(input_data)[0]


    # weather = get_weather(location)
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