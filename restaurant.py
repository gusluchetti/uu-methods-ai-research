import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random

globals()["extra_requirement"] = ""
restaurants = pd.read_csv(r"C:\Users\niels\restaurant_info(1).csv")
food_quality = ["bad food", "okay food", "good food"]
crowdedness = ["not busy", "busy"]
length_of_stay = ["short", "normal", "long"]

random_food_quality = []
random_crowdedness = []
random_length_of_stay = []
for i in range(0, len(restaurants)):
    random_food_quality.append(random.choice(food_quality))
for i in range(0, len(restaurants)):
    random_crowdedness.append(random.choice(crowdedness))
for i in range(0, len(restaurants)):
    random_length_of_stay.append(random.choice(length_of_stay))

restaurants.insert(1, "food_quality", random_food_quality, True)
restaurants.insert(1, "crowdedness", random_crowdedness, True)
restaurants.insert(1, "length_of_stay", random_length_of_stay, True)
restaurants.head()


# In[60]:


restaurants.loc[
    restaurants["length_of_stay"].str.match("(long|short)") & restaurants["area"]
    == "cventer"
]
# restaurants.loc[(restaurants['length_of_stay']== 'long' & restaurants["area"] == 'centre')]


# In[65]:


def find_all_restaurants(df, preferences):
    if len(preferences) == 3:
        if preferences[0] == "any":
            result = df.loc[
                (df["area"] == preferences[1]) & (df["food"] == preferences[2])
            ]
        else:
            result = df.loc[
                (df["pricerange"] == preferences[0])
                & (df["area"] == preferences[1])
                & (df["food"] == preferences[2])
            ]

        result.reset_index(drop=True, inplace=True)
        globals()["remaining_restaurants"] = result
        return result

    elif len(preferences) == 4:
        globals()["extra_requirement"] = preferences[3]
        if preferences[0] == "any":
            pricerange = r"cheap|moderate|expensive"
        else:
            pricerange = preferences[0]
            # df['pricerange'].str.match(pricerange)
            # (lambda x: x['pricerange'].str.contains(pricerange, regex = True))
        if preferences[3] == "romantic":
            result = df.loc[
                df["pricerange"].str.match(pricerange)
                & (df["area"] == preferences[1])
                & (df["food"] == preferences[2])
                & (df["length_of_stay"] == "long")
                & (df["crowdedness"] == "not busy")
            ]

        if preferences[3] == "children":
            result = df.loc[
                df["pricerange"].str.match(pricerange)
                & (df["area"] == preferences[1])
                & (df["food"] == preferences[2])
                & (df["length_of_stay"] == "short")
            ]

        if preferences[3] == "assigned seats":
            result = df.loc[
                df["pricerange"].str.match(pricerange)
                & (df["area"] == preferences[1])
                & (df["food"] == preferences[2])
                & (df["crowdedness"] == "busy")
            ]

        if preferences[3] == "touristic":
            result = df.loc[
                df["pricerange"].str.match(pricerange)
                & (df["area"] == preferences[1])
                & (df["food"] == preferences[2])
                & (df["crowdedness"] == "busy")
                & (df["food"] != "romanian")
                & df["food_quality"]
                == "good food"
            ]

    if len(result) > 0:
        result.reset_index(drop=True, inplace=True)
        globals()["remaining_restaurants"] = result
        return result


def get_extra_requirement():
    if (globals()["extra_requirement"]) == "romantic":
        return (
            " The restaurant is romantic because it allows you to stay for a long time."
        )
    elif globals()["extra_requirement"] == "children":
        return " The restaurant is suitable for children, because the stay is short."
    elif globals()["extra_requirement"] == "assigned seats":
        return " There are assigned seats in this restaurant, because it is busy."
    elif globals()["extra_requirement"] == "touristic":
        return (
            " This restaurant is touristic, because it is cheap and serves good food."
        )


# In[26]:


def toString_recommended_restaurant(index):
    phone = globals()["remaining_restaurants"].iloc[index]["phone"]
    postcode = globals()["remaining_restaurants"].iloc[index]["postcode"]
    rest = globals()["remaining_restaurants"].iloc[index]

    string_no_phone_and_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}."
    string_no_phone = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']})."
    string_no_postcode = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']}. The phonenumber of the restaurant is: {rest['phone']}."
    string_all = f"We recommend the restaurant {rest['restaurantname']}. It is a {rest['pricerange']} priced restaurant that serves {rest['food']}. It is located in the {rest['area']} on {rest['addr']} ({rest['postcode']}). The phonenumber of the restaurant is: {rest['phone']}."

    if (len(globals()["remaining_restaurants"]) > 0) and (
        globals()["extra_requirement"] == ""
    ):
        if (pd.isna(phone)) and (pd.isna(postcode)):
            return string_no_phone_and_postcode
        elif pd.isna(phone):
            return string_no_phone
        elif pd.isna(postcode):
            return string_no_postcode
        else:
            return string_all
    elif globals()["extra_requirement"] != "":
        get_extra_requirement()
        if (pd.isna(phone)) and (pd.isna(postcode)):
            return string_no_phone_and_postcode + get_extra_requirement()
        elif pd.isna(phone):
            return string_no_phone + get_extra_requirement()
        elif pd.isna(postcode):
            return string_no_postcode + get_extra_requirement()
        else:
            return string_all + get_extra_requirement()

    return "Sorry, there are no restaurants that conforms to your requirements."


# In[27]:


if len(globals()["remaining_restaurants"]) >= 1:
    index = np.random.choice(len(globals()["remaining_restaurants"].index))
    print(toString_recommended_restaurant(index))
    globals()["remaining_restaurants"].drop(index, axis=0, inplace=True)
    globals()["remaining_restaurants"].reset_index(drop=True, inplace=True)


# In[ ]:
