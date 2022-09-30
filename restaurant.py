import pandas as pd
import numpy as np
import random
import logging

log = logging.getLogger(__name__)

extra_preference = ""
recommendations = pd.DataFrame([])
restaurants = pd.read_csv(r"datasets/restaurant_info.csv")

food_quality = ["bad food", "okay food", "good food"]
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
    global restaurants, recommendations
    inference_rules = {
        "touristic": "pricerange in ('cheap') & food_quality in ('good') & food not in ('romanian')",
        "assigned seats": "crowdedness in ('busy')",
        "children": "length_of_stay in ('long')",
        "romantic": "crowdedness not in ('busy') & length_of_stay in ('long')",
    }
    # log.debug(f"All restaurants:\n{restaurants}")
    log.debug(f"Current preferences: {form}")

    filter = tuple()
    for key in form.keys():
        log.debug(f"k=({key}) - v=({form[key]})")
        if form[key] == "" or form[key] == "any":
            continue
        if key == "extra_preference":
            temp = inference_rules[form[key]]
        else:
            temp = f"{key} in ('{form[key]}')"
        log.debug(f"filter string: {filter}")
        filter += (temp,)

    query = " & ".join(filter)
    log.debug(f"query: {query}")

    result = restaurants.query(query)
    # result.reset_index(drop=True, inplace=True)
    log.debug(f"Results:\n {result}")
    recommendations = result


def get_recommendations():
    global recommendations
    return recommendations


def drop_recommendation(index):
    global recommendation
    recommendations.drop(index, axis=0, inplace=True)


def get_recommendations_message():
    if len(recommendations) > 0:
        index = np.random.choice(recommendations.index.tolist())
        # recommendations.drop(index, axis=0, inplace=True)
        # recommendations.reset_index(drop=True, inplace=True)
        return recommendation_message(index)
    else:
        return recommendation_message(-1)


def get_extra_requirement():
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
        phone = restaurants.iloc[index]["phone"]
        postcode = restaurants.iloc[index]["postcode"]
        rest = restaurants.iloc[index]

        string_no_phone_and_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}."
        string_no_phone = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']})."
        string_no_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}. The phonenumber of the restaurant is: {rest['phone']}."
        string_all = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']}). The phonenumber of the restaurant is: {rest['phone']}."
        response = ""
        if extra_preference == "":
            if (pd.isna(phone)) and (pd.isna(postcode)):
                response = string_no_phone_and_postcode
            elif pd.isna(phone):
                response = string_no_phone
            elif pd.isna(postcode):
                response = string_no_postcode
            else:
                response = string_all
        elif extra_preference != "":
            response + get_extra_requirement()

        return (index, response + "\nIs that ok?")

    return (
        index,
        "Sorry, there are no restaurants that conforms to your requirements.",
    )
