import logging
import type_match_ls
import restaurant


traversal_tree = {
    "welcome": ["welcome", "test_cuisine"],
    "test_cuisine": ["ask_cuisine", "test_area"],
    "ask_cuisine": ["ask_cuisine", "test_cuisine"],
    "test_area": ["ask_area", "test_price_range"],
    "ask_area": ["ask_area", "test_area"],
    "test_price_range": ["ask_price_range", "confirm_choice"],
    "ask_price_range": ["ask_price_range", "test_price_range"],
    "confirm_choice": ["welcome", "suggest_restaurant"],
    "suggest_restaurant": ["suggest_restaurant", "welcome", "goodbye"],
}
# TODO: add optional sys_dialog for failing conditions
nodes_exec = {
    "welcome": {
        "mode": "welcome",
        "sys_utt": "Hello! What would you like to eat?\n",
        "conditions": ['label!="inform"', "True"],
    },
    "test_cuisine": {
        "mode": "test",
        "sys_utt": "",
        "conditions": ['not get_form("cuisine")', "True"],
    },
    "ask_cuisine": {
        "mode": "extract_cuisine",
        "sys_utt": "What type of food would you like to eat?\n",
        "conditions": ['not get_form("cuisine")', "True"],
    },
    "test_area": {
        "mode": "test",
        "sys_utt": "",
        "conditions": ['not get_form("area")', "True"],
    },
    "ask_area": {
        "mode": "extract_area",
        "sys_utt": "Where would you like to eat?\n",
        "conditions": ['not get_form("area")', "True"],
    },
    "test_price_range": {
        "mode": "test",
        "sys_utt": "",
        "conditions": ['not get_form("price_range")', "True"],
    },
    "ask_price_range": {
        "mode": "extract_price_range",
        "sys_utt": "How pricy you want the food to be?\n",
        "conditions": ['not get_form("price_range")', "True"],
    },
    "confirm_choice": {
        "mode": "confirm",
        "sys_utt": "You would like to eat {}, {} food in {}, correct?\n",
        "conditions": ["label in ['negate','deny']", "True"],
    },
    "suggest_restaurant": {
        "mode": "suggest",
        "sys_utt": "Would you like to eat there: {}\n",
        "conditions": [
            "label in ['negate','deny'] and len(suggestions)>0",
            "len(suggestions)==0",
            "True",
        ],
    },
}

# starting states
current_node = "welcome"
form = {"area": "", "cuisine": "", "price_range": ""}
suggestions = []


def extract_cuisine(utt):
    return type_match_ls.extract_food(utt)


def extract_area(utt):
    return type_match_ls.extract_area(utt)


def extract_price_range(utt):
    return type_match_ls.extract_pricerange(utt)


def reasoning_filter(extra_preferences, restaurant_df):
    """
    args:
      extra_preferences - list of extra preferences
      restaurant_df - dataframe with restaurants and their qualities
    returns: dataframe with restaurants that satisfy inference rules for all given extra_preferences
    """
    inference_rules = {
        "touristic": '(df["pricerange"] == "cheap") & (df["food_quality"] == "good") & (df["food"] != "romanian")',
        "assigned seats": '(df["crowdedness"] == "busy")',
        "children": '(df["length_of_stay"] != "long")',
        "romantic": '(df["crowdedness"] != "busy") & (df["length_of_stay"] == "long")',
    }

    super_rule = " and ".join([inference_rules[x] for x in extra_preferences])
    return restaurant_df.loc[eval(super_rule)]


# get, set and reset form state
def get_form(field):
    global form
    return form[field]


def set_form(field, input):
    global form
    form[field] = input


def reset_form():
    global form
    form = {field: "" for field in form}


def set_suggestions():
    global suggestions
    suggestions = [
        "ChopChop, NY, 5$",
        "TratoriaVerona, Napoli, 40$",
        "Tzaziki, Athens, 1$",
    ]


def set_current_node(new_node):
    global current_node
    current_node = new_node


def traverse(mode, sys_utt, conditions):
    global form
    """Traversing utterance to update form states"""
    logging.debug(f"\nStart: Utterance: {sys_utt}")
    logging.debug(f"Current mode -> {mode}")
    logging.debug(f"Conditions -> {conditions}")

    mode_split = mode.split("_", 1)[0]
    logging.debug(f"mode split? {mode_split}")
    if mode_split in ["ask", "extract", "welcome"]:
        user_utt = input(sys_utt).lower()
        label = classifier(user_utt)
        logging.debug(f"classified utterance as {label}")

        if mode == "welcome":
            reset_form()
            set_form("cuisine", extract_cuisine(user_utt))
            set_form("area", extract_area(user_utt))
            set_form("price_range", extract_price_range(user_utt))
            logging.debug(f"Forms -> {form}")
        elif "extract" in mode:
            field = eval("extract_{}(user_utt)".format(mode.split("_", 1)[1]))
            set_form(mode.split("_", 1)[1], field)

    if mode == "suggest":
        global suggestions
        if len(suggestions) == 0:
            set_suggestions()
        if len(suggestions) > 0:
            suggestion = suggestions.pop(0)
            user_utt = input(sys_utt.format(suggestion)).lower()
            label = classifier(user_utt)
        else:
            print("Sorry. I couldn't find appropriate restaurant.")
    elif mode == "confirm":
        user_utt = input(
            sys_utt.format(
                get_form("price_range"), get_form("cuisine"), get_form("area")
            )
        ).lower()
        label = classifier(user_utt)

    for i, condition in enumerate(conditions):
        logging.debug(f"condition {i}: {condition}")
        logging.debug(f"eval: {eval(condition)}")

        if eval(condition):
            next_node = traversal_tree[current_node][i]
            break

        set_current_node(next_node)


# passing functions that return predictions for the bot to use
def bot(list_models):
    global classifier
    print("Hello! I'm a restaurant recommendation bot! \n")
    classifier_key = input(
        """
Please select a classification method (first two are baseline systems):
[1] - Majority Class
[2 (Default)] Keyword Matching
[3] - Logistic Regression
[4] - Multinomial Naive-Bayes\n"""
    )

    classifier = list_models["2"]
    if classifier_key in list_models.keys():
        classifier = list_models[classifier_key]
        print(f"Model number {classifier_key} was selected!")
    else:
        print("Using default model (keyword matching)")

    global current_node
    while current_node != "goodbye":
        # ** = all args from nodes_exec
        traverse(**nodes_exec[current_node])
        if current_node == "goodbye":
            print("Goodbye!")
