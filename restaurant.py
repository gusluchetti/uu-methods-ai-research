import pandas as pd
import random
import logging

log = logging.getLogger(__name__)


restaurants = pd.read_csv(r"restaurant_info.csv")
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


def find_all_restaurants(df, prefs):
    log.debug(f"All restaurants:\n{df}")
    log.debug(f"Current preferences: {prefs}")

    temp = tuple()
    for key in prefs.keys():
        log.debug(f"k=({key}) - v=({prefs[key]})")
        if prefs[key] == "" or prefs[key] == "unknown":
            continue

        filter = f"{key} in ('{prefs[key]}')"
        log.debug(f"filter string: {filter}")
        temp += (filter,)

    query = " & ".join(temp)
    log.debug(f"query: {query}")

    result = df.query(query)
    result.reset_index(drop=True, inplace=True)
    print(f"These are the restaurants that match your preferences!\n{result}")

    globals()["remaining_restaurants"] = result
    if result.empty:
        print(
            "Sorry, I could not find any restaurant that conforms to your preferences."
        )

    return result


def toString_recommended_restaurant(index):
    if len(globals()["remaining_restaurants"]) > 0:
        phone = globals()["remaining_restaurants"].iloc[index]["phone"]
        postcode = globals()["remaining_restaurants"].iloc[index]["postcode"]
        rest = globals()["remaining_restaurants"].iloc[index]

        string_no_phone_and_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}."
        string_no_phone = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']})."
        string_no_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}. The phonenumber of the restaurant is: {rest['phone']}."
        string_all = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']}). The phonenumber of the restaurant is: {rest['phone']}."

        if (pd.isna(phone)) and (pd.isna(postcode)):
            return string_no_phone_and_postcode
        elif pd.isna(phone):
            return string_no_phone
        elif pd.isna(postcode):
            return string_no_postcode
        else:
            return string_all
    return " "


# globals()["remaining_restaurants"].drop(index, axis=0, inplace=True)
# index = np.random.choice(len(globals()["remaining_restaurants"].index))
# print(toString_recommended_restaurant(index))
# if len(globals()["remaining_restaurants"]) >= 1:
# globals()["remaining_restaurants"].drop(index, axis=0, inplace=True)

# find_all_restaurants(
#    pd.read_csv(r"restaurant_info.csv"),
#    ["moderate", "centre", "british"],
# )
# index = np.random.choice(len(globals()["remaining_restaurants"].index))
