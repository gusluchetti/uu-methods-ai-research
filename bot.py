import logging
from pick import pick

# local imports
import type_match_ls
import restaurant

log = logging.getLogger(__name__)


def create_settings_dict():
    # TODO: validate all these settings (implement hardest)
    # leven_edit - edit levenshtein distance for preference extraction, should be a different menu maybe?
    # confirm_leven - enable confirmation of correctness for levenshtein distance matches, what does that mean?
    # fancy_bot - does fancy bot mean the bot accepts fancy phrases from the user? or that the bot is fancier?
    # stupid_bot - insert artificial errors in preference extraction
    return {
        "enable_restart": {
            "description": "Enable being able to restart the dialogue at any moment",
            "is_enabled": False
        },
        "delayed": {
             "description": "Introduce a delay before showing system responses",
             "is_enabled": False
        },
        "thorough": {
             "description": "Enable confirmation for each preference",
             "is_enabled": False
        },
        "loud": {
             "description": "OUTPUT IN ALL CAPS!!",
             "is_enabled": False
        },
        "voice_assistant": {
             "description": "Enable text-to-speech for system utterances",
             "is_enabled": False
        }
    }


# combining sklearn models with our own baseline systems
# enable user to select any when starting
def create_models_dict(models):
    return {
        "keyword_match": {
            "description": "Matches keyword in utterances to classify dialogue",
            "function": models["1"]
        },
        "majority_class": {
            "description": "Label is always majority class of dataset",
            "function": models["2"]
        },
        "logistic_regression": {
            "description": "Classifies utterance according to fit Logistic Regression model",
            "function": models["3"]
        },
        "multinomial_nb": {
            "description": "Classifies utterance accornding to fit Multinomial Naive-Bayes model",
            "function": models["4"]
        },
    }


# TODO: add optional sys_dialog for failing conditions
# we should alawys get out of the welcome node right?
dialog_tree = {
    "welcome": {
        "mode": "welcome",
        "sys_utt": "Hello! What kind of restaurant are you looking for?\n",
        "exits": ["test_food"],
        "exit_conditions": ["True"],
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
        "exit_conditions": [
            "label in ['negate','deny'] or get_form('extra_preference')",
            "True",
        ],
    },
    "confirm_choice": {
        "mode": "confirm",
        "sys_utt": "",
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
    },
}

# add capability to exit at each point in conversation
for value in dialog_tree.values():
    if value["mode"] != "test":
        value["exits"].insert(0, "goodbye")
        value["exit_conditions"].insert(0, 'label=="bye"')


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


def extract_food(utt):
    return type_match_ls.extract_food(utt)


def extract_area(utt):
    return type_match_ls.extract_area(utt)


def extract_pricerange(utt):
    return type_match_ls.extract_pricerange(utt)


def extract_extra_preference(utt):
    return type_match_ls.extract_extra_preference(utt)


def set_current_node(new_node):
    global current_node
    current_node = new_node


def enable_settings(settings_dict, selected):
    log.debug(f"Selected options: {selected}")
    for s in selected:
        setting = settings_dict[s[1]]["key"]
        if setting == "enable_restart":
            for value in dialog_tree.values():
                if value["mode"] != "test":
                    value["exits"].insert(0, "welcome")
                    value["exit_conditions"].insert(0, '"restart" in user_utt')
        if setting == "loud":
            # TODO: upper all sys utts
            print('loud')


def enable_method(models_dict, selected):
    global classifier

    log.debug(f"Selected model: {selected}")
    classifier = models_dict["2"]


def show_options_menu(options, title, is_multi_select=False, min_multi=0):
    s_list = []
    for k, v in options.items():
        desc = v["description"]
        s_list.append(f"{k} - {desc}")

    # return is list of tuples
    return (pick(
        options=s_list,
        title=title,
        multiselect=is_multi_select,
        min_selection_count=min_multi
    ))


# starting states
current_node = "welcome"
form = {"pricerange": "", "area": "", "food": "", "extra_preference": ""}


# passing functions that return predictions for the bot to use
def start(models_dict):
    global classifier

    settings_dict = create_settings_dict()
    selected_settings = show_options_menu(settings_dict, "Configure your desired settings", True)
    enable_settings(settings_dict, selected_settings)

    selected_method = show_options_menu(models_dict, "Select your classification model")
    enable_method(models_dict, selected_method)

#     classifier_key = input(
#         """
# Please select a classification method (first two are baseline systems):
# [1] - Majority Class
# [2 (Default)] Keyword Matching
# [3] - Logistic Regression
# [4] - Multinomial Naive-Bayes\n"""
#     )
#     classifier = list_models["2"]
#     if classifier_key in list_models.keys():
#         classifier = list_models[classifier_key]
#         print(f"Using model {classifier_key}")
#     else:
#         print("Using default model (keyword matching)")

    global current_node
    while current_node != "goodbye":
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
                field = eval(f"extract_{mode_split[1]}(user_utt)")
                set_form(mode_split[1], field)

        if mode == "confirm":
            area = get_form("area")
            food = get_form("food")
            pricerange = get_form("pricerange")

            start = "You want to eat in a restaurant"
            end = ", did I get that right?\n"
            extras = ""

            if food != "any" or "":
                extras += f" that has {food} cuisine"
            if area != "any" or "":
                extras += f" somewhere around the {area}"
            if pricerange != "any" or "":
                extras += f" in the {pricerange} price range"

            none = "All right! A surprise it is. Is that ok?\n"
            if food == "any" and area == "any" and pricerange == "any":
                sys_utt = none
            else:
                sys_utt = f"{start}{extras}{end}"

            user_utt = input(sys_utt)
            label = classifier(user_utt)
            log.debug(f"classified utterance as {label}")
            restaurant.set_recommendations(form)

        elif mode == "suggest":
            recommendations = restaurant.get_recommendations()
            index, sys_utt = restaurant.get_recommendations_message()
            sys_utt = f"Found {len(recommendations)} matching restaurant(s).\n" + sys_utt

            if recommendations.empty:
                print(sys_utt)
                label = "null"
            else:
                user_utt = input(sys_utt).lower()
                label = classifier(user_utt)
                log.debug(f"classified utterance as {label}")
                restaurant.drop_recommendation(index)

        # evaluate exit conditions
        for i, condition in enumerate(conditions):
            log.debug(f"condition {i}: {condition} is evaluated as {eval(condition)}")
            if eval(condition):
                next_node = exits[i]
                break

        set_current_node(next_node)
    print("Goodbye! Thanks for using our bot!")
