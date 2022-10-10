import logging

# local imports
import type_match_ls as ls
import restaurant
import lib

log = logging.getLogger(__name__)


def reset_form():
    global form
    form = {field: "" for field in form}


# any setting that is related to the dialog transition system
# goes here
def enable_settings(settings_dict, selected):
    global dialog_tree
    log.debug(f"Selected options: {selected}")
    # getting option key based on index
    for s in selected:
        setting = list(settings_dict)[s[1]]
        for k, v in dialog_tree.items():
            if setting == "enable_restart" and v["mode"] != "test":
                v["exits"].insert(0, "welcome")
                v["exit_conditions"].insert(0, '"restart" in user_utt')
            if setting == "loud":
                v["sys_utt"] = v["sys_utt"].upper()


# called whenever user response is necessary
def get_user_input() -> str:
    return "test"


# global variables
dialog_tree = None
classifier = None
current_node = "welcome"
form = {"food": "", "pricerange": "", "area": "", "extra_preference": ""}


def start(list_models):
    global classifier, dialog_tree, current_node, classifier, form
    dialog_tree = lib.create_dialog_tree()

    # showing configurability menu
    settings_dict = lib.create_settings_dict()
    selected_settings = lib.show_options_menu(settings_dict, "Configure your desired settings", True)
    enable_settings(settings_dict, selected_settings)
    # showing model selection menu
    models_dict = lib.create_models_dict(list_models)
    selected_method = lib.show_options_menu(models_dict, "Select your classification model")
    classifier = lib.enable_method(models_dict, selected_method)
    log.debug(classifier)

    while current_node != "goodbye":
        mode = dialog_tree[current_node]["mode"]
        sys_utt = dialog_tree[current_node]["sys_utt"]
        exits = dialog_tree[current_node]["exits"]
        conditions = dialog_tree[current_node]["exit_conditions"]
        log.debug(f"\nCurrent Node: {current_node}\nMode: {mode}")
        log.debug(f"\nExits: {exits}\nConditions: {conditions}\nForm: {form}")

        mode_split = mode.split("_", maxsplit=1)
        if mode_split[0] in ["ask", "extract", "welcome"]:
            user_utt = input(sys_utt).lower()
            label = classifier(user_utt)
            log.debug(f"classified utterance as {label}")

            if mode == "welcome":
                reset_form()
                form["food"] = ls.extract_food(user_utt)
                form["area"] = ls.extract_area(user_utt)
                form["pricerange"] = ls.extract_pricerange(user_utt)
                form["extra_preference"] = ls.extract_extra_preference(user_utt)
            elif "extract" in mode:
                field = eval(f"ls.extract_{mode_split[1]}(user_utt)")
                form[mode_split[1]] = field

        if mode == "confirm":
            sys_utt = lib.get_confirmation_msg(form)
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
            evaluation = eval(condition)
            log.debug(f"condition {i}: {condition} is {evaluation}")

            if evaluation:
                next_node = exits[i]
                break

        current_node = next_node
    print("Goodbye! Thanks for using our bot!")
