import pandas as pd
import numpy as np
import random

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
restaurants.head()


def find_all_restaurants(df, preferences):
    if len(preferences) == 3:
        result = df.loc[
            (df["pricerange"] == preferences[0])
            & (df["area"] == preferences[1])
            & (df["food"] == preferences[2])
        ]
        result.reset_index(drop=True, inplace=True)
        globals()["remaining_restaurants"] = result
        return result
    elif len(preferences) == 6:
        result = df.loc[
            (df["pricerange"] == preferences[0])
            & (df["area"] == preferences[1])
            & (df["food"] == preferences[2])
            & (df["length_of_stay"] == preferences[3])
            & (df["crowdedness"] == preferences[4])
            & (df["food_quality"] == preferences[5])
        ]
        result.reset_index(drop=True, inplace=True)
        globals()["remaining_restaurants"] = result
        return result
    else:
        return "Sorry, there is no restaurant that conforms to you requirements."


find_all_restaurants(
    pd.read_csv(r"restaurant_info.csv"),
    ["moderate", "centre", "british"],
)


index = np.random.choice(len(globals()["remaining_restaurants"].index))


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
