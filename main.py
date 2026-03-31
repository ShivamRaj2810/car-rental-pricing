from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import random
import sqlite3
import requests
import os
import time

from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from pricing_logic import surge_price, weather_adjustment

# -------------------------------
# ENV
# -------------------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"

# -------------------------------
# MODELS
# -------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class PriceRequest(BaseModel):
    pickup: str
    drop: str
    base_price: float

# -------------------------------
# APP
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for local + frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# AUTH
# -------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------------
# DATABASE
# -------------------------------
def create_user_table():
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT,
        customer_rating REAL DEFAULT 4.0
    )
    """)

    conn.commit()
    conn.close()

create_user_table()

def get_user(email: str):
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT email, password, customer_rating FROM users WHERE email=?", (email,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return {"email": row[0], "password": row[1], "customer_rating": row[2]}
    return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")

        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = get_user(email)

        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------
# REGISTER
# -------------------------------
@app.post("/register")
def register_user(user: UserRegister):

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email=?", (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = get_password_hash(user.password)

    cursor.execute(
        "INSERT INTO users (email, password, customer_rating) VALUES (?, ?, ?)",
        (user.email, hashed_password, 4.0)
    )

    conn.commit()
    conn.close()

    return {"message": "User registered successfully ✅"}

# -------------------------------
# LOGIN
# -------------------------------
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)

    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token({"sub": user["email"]})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# -------------------------------
# UTILS
# -------------------------------
def get_weekend():
    return 1 if datetime.now().weekday() >= 5 else 0

def get_available_cars(location):
    location = location.lower()

    if "airport" in location:
        return random.randint(10, 30)
    elif "downtown" in location:
        return random.randint(20, 50)
    else:
        return random.randint(40, 80)

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        res = requests.get(url)
        data = res.json()

        if data.get("cod") != 200:
            return "clear"

        w = data["weather"][0]["main"].lower()

        if "rain" in w:
            return "rain"
        elif "storm" in w:
            return "storm"
        elif "cloud" in w:
            return "cloudy"
        else:
            return "clear"

    except:
        return "clear"

# -------------------------------
# OSRM DISTANCE
# -------------------------------
def get_coordinates(place):
    try:
        url = "https://nominatim.openstreetmap.org/search"

        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "car-rental-app"}

        res = requests.get(url, params=params, headers=headers)
        data = res.json()

        if not data:
            return None, None

        return float(data[0]["lat"]), float(data[0]["lon"])

    except:
        return None, None


def get_distance(pickup, drop):
    try:
        lat1, lon1 = get_coordinates(pickup)
        time.sleep(1)  # prevent rate limit
        lat2, lon2 = get_coordinates(drop)

        if not lat1 or not lat2:
            return None, "Invalid location"

        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

        res = requests.get(url)
        data = res.json()

        if "routes" not in data:
            return None, "Route not found"

        distance_km = data["routes"][0]["distance"] / 1000

        return distance_km, f"{distance_km:.2f} km"

    except:
        return None, "Distance error"

# -------------------------------
# LOAD MODEL
# -------------------------------
price_model = pickle.load(open("models/price_model.pkl", "rb"))

def get_predicted_demand():
    df = pd.read_csv("dataset/demand_data.csv")
    recent = df["bookings"].tail(10)
    weights = np.arange(1, len(recent) + 1)
    return float(np.average(recent, weights=weights))

# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict_price")
def predict_price(data: PriceRequest, current_user: dict = Depends(get_current_user)):

    pickup = data.pickup.strip()
    drop = data.drop.strip()
    base_price = data.base_price

    distance_km, distance_text = get_distance(pickup, drop)

    predicted_demand = get_predicted_demand()

    # 🔥 fallback (VERY IMPORTANT)
    if distance_km is None:
        print("Using fallback distance")
        distance_km = random.uniform(3, 10)
        distance_text = f"{distance_km:.2f} km (approx)"

    available_cars = get_available_cars(pickup)
    weekend = get_weekend()
    rating = current_user["customer_rating"]
    weather = get_weather(pickup)

    demand = get_predicted_demand()

    # price logic
    distance_cost = distance_km * 12
    base_price += distance_cost

    price = surge_price(base_price, demand, available_cars)
    price = weather_adjustment(price, weather)

    input_data = pd.DataFrame([{
        "demand": demand,
        "available_cars": available_cars,
        "weekend": weekend,
        "customer_rating": rating,
        "base_price": price
    }])

    final_price = price_model.predict(input_data)[0]

    return {
    "pickup": pickup,
    "drop": drop,
    "distance_km": float(distance_km),
    "distance_text": distance_text,   # 🔥 THIS WAS MISSING / FAILING
    "distance_cost": float(distance_cost),
    "dynamic_price": float(final_price),
    "demand": float(predicted_demand),
    "weather": weather
}

# -------------------------------
# BOOKING
# -------------------------------
@app.post("/book")
def book(current_user: dict = Depends(get_current_user)):

    email = current_user["email"]
    rating = min(5.0, current_user["customer_rating"] + 0.1)

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE users SET customer_rating=? WHERE email=?",
        (rating, email)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Ride booked 🚗",
        "new_rating": rating
    }