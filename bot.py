# local imports
import type_match_ls
import restaurant

import logging

log = logging.getLogger(__name__)


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
        "exits": ["confirm_choice", "ask_extra_preference"],
        "exit_conditions": ["label in ['negate','deny'] or get_form('extra_preference')", "True"],
    },
    "confirm_choice": {
        "mode": "confirm",
        "sys_utt": "You would like to eat {}, {} food in {}, correct?\n",
        "exits": ["welcome", "suggest_restaurant"],
        "exit_conditions": ["label in ['negate','deny']", "True"],
    },
    "suggest_restaurant": {
        "mode": "suggest",
        "sys_utt": "",
        "exits": ["suggest_restaurant", "goodbye", "goodbye"],
        "exit_conditions": [
            "label in ['negate','deny']",
            "restaurant.get_recommendations().empty",
            "True",
        ],
    }
}



# add capability to exit at each point in conversation
for value in dialog_tree.values():
    if value["mode"] != "test":
        value["exits"].insert(0, "goodbye")
        value["exit_conditions"].insert(0, 'label=="bye"')

# starting states
current_node = "welcome"
form = {"pricerange": "", "area": "", "food": "", "extra_preference": ""}


def extract_food(utt):
    return type_match_ls.extract_food(utt)


def extract_area(utt):
    return type_match_ls.extract_area(utt)


def extract_pricerange(utt):
    return type_match_ls.extract_pricerange(utt)


def extract_extra_preference(utt):
    return type_match_ls.extract_extra_preference(utt)


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


def set_current_node(new_node):
    global current_node
    current_node = new_node


def traverse_dialog_tree(current_node):
    mode = dialog_tree[current_node]["mode"]
    sys_utt = dialog_tree[current_node]["sys_utt"]
    exits = dialog_tree[current_node]["exits"]
    conditions = dialog_tree[current_node]["exit_conditions"]

    global form
    """Traversing utterance to update form states"""
    log.debug(
        f"\nCurrent Node: {current_node}\nMode: {mode}\nExits: {exits}\nConditions: {conditions}\nForm: {form}"
    )

    mode_split = mode.split("_", 1)
    if mode_split[0] in ["ask", "extract", "welcome"]:
        user_utt = input(sys_utt).lower()
        label = classifier(user_utt)
        log.debug(f"classified utterance as {label}")

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
        index, suggestion = restaurant.get_recommendations_message()
        log.debug(f"suggestions {suggestion}")
        if restaurant.get_recommendations().empty:
            print(suggestion)
            label = "null"
        else:
            user_utt = input(suggestion).lower()
            label = classifier(user_utt)
            log.debug(f"classified utterance as {label}")
            restaurant.drop_recommendation(index)

    elif mode == "confirm":
        user_utt = input(
            sys_utt.format(get_form("pricerange"), get_form("food"), get_form("area"))
        ).lower()
        label = classifier(user_utt)
        log.debug(f"classified utterance as {label}")
        restaurant.set_recommendations(form)

    for i, condition in enumerate(conditions):
        log.debug(f"condition {i}: {condition} is evaluated as {eval(condition)}")
        if eval(condition):
            next_node = exits[i]
            break

    set_current_node(next_node)


# passing functions that return predictions for the bot to use
def start(list_models):
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
    flag = input("Do you want to be able to restart conversation? [y/n]\n")
    if flag == "y":
        for value in dialog_tree.values():
            if value["mode"] != "test":
                value['exits'].insert(0, 'welcome')
                value['exit_conditions'].insert(0, '"restart" in user_utt')
        
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
