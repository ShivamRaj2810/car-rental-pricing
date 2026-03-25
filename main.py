from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from datetime import datetime
import random
import sqlite3
import requests
import os

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

# -------------------------------
# USER MODELS
# -------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str

# -------------------------------
# APP
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# JWT CONFIG
# -------------------------------
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

# -------------------------------
# AUTH HELPERS
# -------------------------------
def get_user(email: str):
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT email, password, customer_rating FROM users WHERE email=?", (email,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return {"email": row[0], "password": row[1], "customer_rating": row[2]}
    return None


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


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

    # check if user exists
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
# LOGIN (JWT)
# -------------------------------
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    print("EMAIL:", form_data.username)
    print("USER:", user)

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
        elif "storm" in w or "thunder" in w:
            return "storm"
        elif "cloud" in w:
            return "cloudy"
        elif "haze" in w or "fog" in w or "mist" in w:
            return "foggy"
        else:
            return "clear"

    except:
        return "clear"


# -------------------------------
# LOAD MODELS
# -------------------------------
price_model = pickle.load(open("models/price_model.pkl", "rb"))
demand_model = load_model("models/demand_lstm.keras", compile=False)


def get_past_demand():
    df = pd.read_csv("dataset/demand_data.csv")
    return df["bookings"].tail(10).values


# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"message": "Dynamic Pricing API Running"}


@app.post("/predict_price")
def predict_price(data: dict, current_user: dict = Depends(get_current_user)):

    location = data["location"]
    base_price = data["base_price"]

    available_cars = get_available_cars(location)
    weekend = get_weekend()
    rating = current_user["customer_rating"]

    weather = get_weather(location)

    # Demand
    past = np.array(get_past_demand()).reshape(1, 10, 1)
    predicted_demand = demand_model.predict(past)[0][0]

    # Pricing
    price = surge_price(base_price, predicted_demand, available_cars)
    price = weather_adjustment(price, weather)

    # ML model
    input_data = pd.DataFrame([{
        "demand": predicted_demand,
        "available_cars": available_cars,
        "weekend": weekend,
        "customer_rating": rating,
        "base_price": price
    }])

    final_price = price_model.predict(input_data)[0]

    return {
        "demand": float(predicted_demand),
        "dynamic_price": float(final_price),
        "available_cars": available_cars,
        "customer_rating": rating,
        "weekend": weekend,
        "weather": weather
    }


# -------------------------------
# BOOKING → UPDATE RATING
# -------------------------------
@app.post("/book")
def book(current_user: dict = Depends(get_current_user)):

    email = current_user["email"]
    current_rating = current_user["customer_rating"]

    new_rating = min(5.0, current_rating + 0.1)

    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE users SET customer_rating=? WHERE email=?",
        (new_rating, email)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Booking successful 🚗",
        "new_rating": new_rating
    }