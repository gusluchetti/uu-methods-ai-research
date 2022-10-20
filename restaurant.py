import pandas as pd
import numpy as np
import random
import logging

log = logging.getLogger(__name__)

extra_preference = ""
recommendations = pd.DataFrame([])
restaurants = pd.read_csv(r"datasets/restaurant_info.csv")

food_quality = ["bad", "okay", "good"]
crowdedness = ["not busy", "busy"]
length_of_stay = ["short", "normal", "long"]

random_food_quality = []
random_crowdedness = []
random_length_of_stay = []
for i in range(0, len(restaurants)):
    random_food_quality.append(random.choice(food_quality))
    random_crowdedness.append(random.choice(crowdedness))
    random_length_of_stay.append(random.choice(length_of_stay))

restaurants.insert(1, "food_quality", random_food_quality, True)
restaurants.insert(1, "crowdedness", random_crowdedness, True)
restaurants.insert(1, "length_of_stay", random_length_of_stay, True)


def set_recommendations(form):
    global restaurants, recommendations, extra_preference
    inference_rules = {
        "touristic": "pricerange in ('cheap') & food_quality in ('good') & food not in ('romanian')",
        "assigned seats": "crowdedness in ('busy')",
        "children": "length_of_stay not in ('long')",
        "romantic": "crowdedness not in ('busy') & length_of_stay in ('long')",
    }
    extra_preference = form["extra_preference"]
    log.debug(f"Current preferences: {form}")

    filter = tuple()
    for key in form.keys():
        log.debug(f"k=({key}) - v=({form[key]})")
        if form[key] == "" or form[key] == "any":
            continue
        if key == "extra_preference":
            temp = inference_rules[form[key]]
            filter += (temp,)
        else:
            temp = f"{key} in ('{form[key]}')"
            filter += (temp,)
        log.debug(f"filter string: {filter}")

    log.debug(f"total restaurants: {restaurants.size}")
    if not filter:
        recommendations = restaurants
    else:
        query = " & ".join(filter)
        log.debug(f"query: {query}")
        result = restaurants.query(query)
        log.debug(f"Results:\n {result}")

        recommendations = result


def get_recommendations():
    global recommendations
    return recommendations


def drop_recommendation(index):
    global recommendations
    dropped = recommendations.copy().drop(index, axis=0)
    recommendations = dropped


def get_recommendations_message():
    if len(recommendations) > 0:
        index = np.random.choice(recommendations.index.tolist())
        return recommendation_message(index)
    else:
        return recommendation_message(-1)


def get_extra_preference_msg():
    global extra_preference
    if extra_preference == "romantic":
        return (
            "The restaurant is romantic because it allows you to stay for a long time."
        )
    elif extra_preference == "children":
        return "The restaurant is suitable for children, because the stay is short."
    elif extra_preference == "assigned seats":
        return "There are assigned seats in this restaurant, because it is busy."
    elif extra_preference == "touristic":
        return "This restaurant is touristic, because it is cheap and serves good food."
    else:
        return ""


def recommendation_message(index):
    if index >= 0:
        restaurant = restaurants.iloc[index]
        name = restaurant["restaurantname"]
        pricerange = restaurant["pricerange"]
        food = restaurant["food"]
        area = restaurant["area"]
        address = restaurant["addr"]
        postcode = restaurant["postcode"]
        phone = restaurant["phone"]

        name_price_food = f"We recommend {name}, a {pricerange} priced restaurant that serves {food} food"
        address = f"It's located in the {area} on {address}"
        postcode = f"({postcode})"
        phone = f"You can call them at {phone}"

        def_msg = f"{name_price_food}. {address}"
        if (pd.isna(phone)) and (pd.isna(postcode)):
            response = f"{def_msg}."
        elif pd.isna(phone):
            response = f"{def_msg}{postcode}."
        elif pd.isna(postcode):
            response = f"{def_msg}. {phone}."
        else:
            response = f"{def_msg}{postcode}. {phone}."

        extra_msg = get_extra_preference_msg()
        response = f"{response} {extra_msg}"
        return (index, response + "\nIs that ok?\n")

    return (
        index,
        "Sorry, there are no restaurants that conforms to your requirements.",
    )
