# local imports
import type_match_ls
import restaurant
import base_logger


# TODO: add optional sys_dialog for failing conditions
dialog_tree = {
    "welcome": {
        "mode": "welcome",
        "sys_utt": "Hello! What would you like to eat?\n",
        "exits": ["welcome", "test_food"],
        "exit_conditions": ['label!="inform"', "True"],
    },
    "test_food": {
        "mode": "test",
        "sys_utt": "",
        "exits": ["ask_food", "test_area"],
        "exit_conditions": ['not get_form("food")', "True"],
    },
    "ask_food": {
        "mode": "extract_food",
        "sys_utt": "What type of food would you like to eat?\n",
        "exits": ["ask_food", "test_food"],
        "exit_conditions": ['not get_form("food")', "True"],
    },
    "test_area": {
        "mode": "test",
        "sys_utt": "",
        "exits": ["ask_area", "test_pricerange"],
        "exit_conditions": ['not get_form("area")', "True"],
    },
    "ask_area": {
        "mode": "extract_area",
        "sys_utt": "Where would you like to eat?\n",
        "exits": ["ask_area", "test_area"],
        "exit_conditions": ['not get_form("area")', "True"],
    },
    "test_pricerange": {
        "mode": "test",
        "sys_utt": "",
        "exits": ["ask_pricerange", "ask_extra_preference"],
        "exit_conditions": ['not get_form("pricerange")', "True"],
    },
    "ask_pricerange": {
        "mode": "extract_pricerange",
        "sys_utt": "How pricy you want the food to be?\n",
        "exits": ["ask_pricerange", "test_pricerange"],
        "exit_conditions": ['not get_form("pricerange")', "True"],
    },
    "ask_extra_preference": {
        "mode": "extract_extra_preference",
        "sys_utt": "Do you have any extra preference?\n",
        "exits": ["ask_extra_preference", "confirm_choice"],
        "exit_conditions": ['not get_form("extra_preference")', "True"],
    },
    "confirm_choice": {
        "mode": "confirm",
        "sys_utt": "You would like to eat {}, {} food in {}, correct?\n",
        "exits": ["welcome", "suggest_restaurant"],
        "exit_conditions": ["label in ['negate','deny']", "True"],
    },
    "suggest_restaurant": {
        "mode": "suggest",
        "sys_utt": "Would you like to eat there: {}\n",
        "exits": ["suggest_restaurant", "welcome", "goodbye"],
        "exit_conditions": [
            "label in ['negate','deny'] and len(suggestions)>0",
            "len(suggestions)==0",
            "True",
        ],
    },
}

# if restart_flag = True:
#     for value in dialog_tree.values():
#       value['exits'].insert(0, 'welcome')
#       value['exit_conditions'].insert(0, '"restart" in user_utt')

# add capability to exit at each point in conversation
for value in dialog_tree.values():
    if value["mode"] != "test":
        value["exits"].insert(0, "goodbye")
        value["exit_conditions"].insert(0, 'label=="bye"')

# starting states
current_node = "welcome"
form = {"area": "", "food": "", "pricerange": "", "extra_preference": ""}
suggestions = []


def extract_food(utt):
    return type_match_ls.extract_food(utt)


def extract_area(utt):
    return type_match_ls.extract_area(utt)


def extract_pricerange(utt):
    return type_match_ls.extract_pricerange(utt)


def extract_extra_preference(utt):
    return type_match_ls.extract_extra_preference(utt)


def reasoning_filter(extra_preference, restaurant_df):
    """
    args:
      extra_preference - extra preference string
      restaurant_df - dataframe with restaurants and their qualities
    returns:
      dataframe with restaurants that satisfy inference rules for all given extra_preferences
    """
    inference_rules = {
        "touristic": '(df["pricerange"] == "cheap") & (df["food_quality"] == "good") & (df["food"] != "romanian")',
        "assigned seats": '(df["crowdedness"] == "busy")',
        "children": '(df["length_of_stay"] != "long")',
        "romantic": '(df["crowdedness"] != "busy") & (df["length_of_stay"] == "long")',
    }

    #  super_rule = " and ".join([inference_rules[x] for x in extra_preferences])
    #  return restaurant_df.loc[eval(super_rule)]
    return restaurant_df.loc[eval(inference_rules[extra_preference])]


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
    suggestions = restaurant.find_all_restaurants(
        restaurant.restaurants,
        [
            form["pricerange"],
            form["area"],
            form["food"],
            form["extra_preference"],
        ],
    )
    logger.debug(f"\n {len(suggestions)} suggestions possible -> {suggestions}")


def set_current_node(new_node):
    global current_node
    current_node = new_node


def traverse_dialog_tree(current_node):
    mode = dialog_tree[current_node]["mode"]
    sys_utt = dialog_tree[current_node]["sys_utt"]
    exits = dialog_tree[current_node]["exits"]
    conditions = dialog_tree[current_node]["exit_conditions"]

    global form, logger
    """Traversing utterance to update form states"""
    logger.debug(
        f"\nCurrent Node: {current_node}\nMode: {mode}\nExits: {exits}\nConditions: {conditions}\nForm: {form}"
    )

    mode_split = mode.split("_", 1)
    if mode_split[0] in ["ask", "extract", "welcome"]:
        user_utt = input(sys_utt).lower()
        label = classifier(user_utt)
        logger.debug(f"classified utterance as {label}")

        if mode == "welcome":
            reset_form()
            set_form("food", extract_food(user_utt))
            set_form("area", extract_area(user_utt))
            set_form("pricerange", extract_pricerange(user_utt))
            set_form("extra_preference", extract_extra_preference(user_utt))
        elif "extract" in mode:
            field = eval("extract_{}(user_utt)".format(mode_split[1]))
            set_form(mode_split[1], field)

    if mode == "suggest":
        global suggestions
        if len(suggestions) <= 0:
            set_suggestions()
        else:
            suggestion = suggestions.pop(0)
            user_utt = input(sys_utt.format(suggestion)).lower()
            label = classifier(user_utt)

        if len(suggestions) == 0:  # if nothing was found...
            print("Sorry. I couldn't find appropriate restaurant.")

    elif mode == "confirm":
        user_utt = input(
            sys_utt.format(get_form("pricerange"), get_form("food"), get_form("area"))
        ).lower()
        label = classifier(user_utt)

    for i, condition in enumerate(conditions):
        logger.debug(f"condition {i}: {condition} is evaluated as {eval(condition)}")
        if eval(condition):
            next_node = exits[i]
            break

    set_current_node(next_node)


# passing functions that return predictions for the bot to use
def start(list_models):
    global logger
    logger = base_logger.get_logger()
    print(logger.name)

    global classifier
    print("\nHello! I'm a restaurant recommendation bot!")
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
    else:
        print("Using default model (keyword matching)")

    global current_node
    while current_node != "goodbye":
        traverse_dialog_tree(current_node)
        if current_node == "goodbye":
            print("Goodbye!")
