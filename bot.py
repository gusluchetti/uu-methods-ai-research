import logging
import simple_term_menu

# local imports
import type_match_ls
import restaurant

log = logging.getLogger(__name__)


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
        "sys_utt": "You would like to eat {get_form(pricerange)}, \
        {get_form(food)} food in the {get_form(area)} part of town, correct?\n",
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
            field = eval(f"extract_{mode_split[1]}(user_utt)")
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
        user_utt = input(f"{sys_utt}")
        label = classifier(user_utt)
        log.debug(f"classified utterance as {label}")
        restaurant.set_recommendations(form)

    # eval exit conditions
    for i, condition in enumerate(conditions):
        log.debug(f"condition {i}: {condition} is evaluated as {eval(condition)}")
        if eval(condition):
            next_node = exits[i]
            break

    set_current_node(next_node)


def create_settings_dict():
    # TODO:
    # leven_edit - edit levenshtein distance for preference extraction, should be a different menu maybe?
    # fancy_bot - does fancy bot mean the bot accepts fancy phrases from the user? or that the bot is fancier?
    settings_dict = [
        {
            "key": "confirm_leven",
            "description": "Enable confirmation of correctness for Levenshtein distance matches",
            "is_enabled": False
        },
        {
            "key": "random_order",
            "description": "Enable preferences to be stated in random order",
            "is_enabled": False
        },
        {
            "key": "stupid_bot",
            "description": "Insert artificial errors in preference extraction",
            "is_enabled": False
        },
        {
            "key": "enable_restart",
            "description": "Enable being able to restart the dialog at any moment",
            "is_enabled": False
        },
        {
            "key": "delayed",
            "description": "Introduce a delay before showing system responses",
            "is_enabled": False
        },
        {
            "key": "thorough",
            "description": "Enable confirmation for each preference",
            "is_enabled": False
        },
        {
            "key": "loud",
            "description": "OUTPUT IN ALL CAPS!!",
            "is_enabled": False
        },
        {
            "key": "voice_assistant",
            "description": "Enable text-to-speech for system utterances",
            "is_enabled": False
        }
    ]
    return settings_dict


# passing functions that return predictions for the bot to use
def start(list_models):
    global classifier
    print("\nHello there! I'm a restaurant recommendation bot!")

    def show_settings_menu(settings_dict):
        s_list = []
        for s in settings_dict:
            key, desc = s["key"], s["description"]
            s_list.append(f"{key} - {desc}")

        settings = simple_term_menu.TerminalMenu(
            s_list,
            multi_select=True,
            multi_select_empty_ok=True,
            show_multi_select_hint=True,
        )

        settings_menu_selected = settings.show()
        return (settings_menu_selected, settings.chosen_menu_entries)

    print("Configure your desired settings:")
    settings_dict = create_settings_dict()
    sms, sm_entries = show_settings_menu(settings_dict)
    log.debug(f"Enabled settings: {sm_entries}")

    for selected in sms:
        setting = settings_dict[selected]["key"]
        if setting == "enable_restart":
            for value in dialog_tree.values():
                if value["mode"] != "test":
                    value["exits"].insert(0, "welcome")
                    value["exit_conditions"].insert(0, '"restart" in user_utt')
        if setting == "loud":
            # upper all sys utts
            print('loud')

    # selection of classification method
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
        print(f"Using model {classifier_key}")
    else:
        print("Using default model (keyword matching)")

    global current_node
    while current_node != "goodbye":
        traverse_dialog_tree(current_node)
        if current_node == "goodbye":
            print("Goodbye!")
