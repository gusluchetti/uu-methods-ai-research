import logging
from pick import pick

log = logging.getLogger(__name__)
# here lies dict creation and other string bulding operations used in bot.py


def create_dialog_tree():
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
            "exit_conditions": ['not form["food"]', "True"],
        },
        "ask_food": {
            "mode": "extract_food",
            "sys_utt": "What type of food would you like to eat?\n",
            "exits": ["ask_food", "test_food"],
            "exit_conditions": ['not form["food"]', "True"],
        },
        "test_area": {
            "mode": "test",
            "sys_utt": "",
            "exits": ["ask_area", "test_pricerange"],
            "exit_conditions": ['not form["food"]', "True"],
        },
        "ask_area": {
            "mode": "extract_area",
            "sys_utt": "Where would you like to eat?\n",
            "exits": ["ask_area", "test_area"],
            "exit_conditions": ['not form["area"]', "True"],
        },
        "test_pricerange": {
            "mode": "test",
            "sys_utt": "",
            "exits": ["ask_pricerange", "ask_extra_preference"],
            "exit_conditions": ['not form["pricerange"]', "True"],
        },
        "ask_pricerange": {
            "mode": "extract_pricerange",
            "sys_utt": "How pricy you want the food to be?\n",
            "exits": ["ask_pricerange", "test_pricerange"],
            "exit_conditions": ['not form["pricerange"]', "True"],
        },
        "ask_extra_preference": {
            "mode": "extract_extra_preference",
            "sys_utt": "Do you have any extra preference?\n",
            "exits": ["confirm_choice", "ask_extra_preference"],
            "exit_conditions": [
                "label in ['negate','deny'] or form['extra_preference']",
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
        }
    }

    # add capability to exit at each point in conversation
    for value in dialog_tree.values():
        if value["mode"] != "test":
            value["exits"].insert(0, "goodbye")
            value["exit_conditions"].insert(0, 'label=="bye"')

    return dialog_tree


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
        "majority_class": {
            "description": "Label is always majority class of dataset",
            "function": models[0]
        },
        "keyword_match": {
            "description": "Matches keyword in utterances to classify dialogue",
            "function": models[1]
        },
        "logistic_regression": {
            "description": "Classifies utterance according to fit Logistic Regression model",
            "function": models[2]
        },
        "multinomial_nb": {
            "description": "Classifies utterance accornding to fit Multinomial Naive-Bayes model",
            "function": models[3]
        }
    }


def enable_method(models_dict, selected):
    global classifier
    key = list(models_dict)[selected[1]]
    log.info(f"You've selected model {key}")

    return models_dict[key]["function"]


def show_options_menu(options, title, is_multi_select=False, min_multi=0):
    log.debug(options)
    s_list = []
    for k, v in options.items():
        print(k, v)
        desc = v["description"]
        s_list.append(f"{k} - {desc}")

    # return is list of tuples
    return (pick(
        options=s_list,
        title=title,
        multiselect=is_multi_select,
        min_selection_count=min_multi
    ))


def get_confirmation_msg(food, area, pricerange) -> str:
    start = "You want to eat in a restaurant"
    end = ", did I get that right?\n"
    extras = ""

    if food != "any" or "":
        extras += f" that has {food} cuisine"
    if area != "any" or "":
        extras += f" somewhere around the {area}"
    if pricerange != "any" or "":
        extras += f" in the {pricerange} price range"

    if food == "any" and area == "any" and pricerange == "any":
        sys_utt = "All right! A surprise it is. Is that ok?\n"
    else:
        sys_utt = f"{start}{extras}{end}"

    return sys_utt
