def surge_price(base_price, demand, supply):

    ratio = demand / supply

    if ratio <= 1:
        multiplier = 1
    elif ratio <= 1.5:
        multiplier = 1.2
    elif ratio <= 2:
        multiplier = 1.5
    else:
        multiplier = 2

    return base_price * multiplier


def event_adjustment(price, attendance):

    multiplier = 1 + (attendance / 100000)

    return price * multiplier


def weather_adjustment(price, weather):

    if weather == "rain":
        price *= 1.1
    elif weather == "storm":
        price *= 1.25

    return price