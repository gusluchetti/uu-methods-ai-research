import logging
import time
import pyttsx3

# local imports
import type_match_ls as ls
import restaurant
import lib

log = logging.getLogger(__name__)
engine = pyttsx3.init()


def reset_form():
    global form
    form = {field: "" for field in form}


def enable_settings(s_dict, selected):
    global dialog_tree
    log.debug(f"Selected options: {selected}")
    for s in selected:
        setting = list(s_dict)[s[1]]
        s_dict[setting]["is_enabled"] = True

    # any setting that modifies aspects of the state system
    for k, v in dialog_tree.items():
        if s_dict["enable_restart"]["is_enabled"] and v["mode"] != "test":
            v["exits"].insert(0, "welcome")
            v["exit_conditions"].insert(0, '"restart" in user_utt')

    return s_dict


# modify sys_utt if any settings call for it
def get_sys_utt(sys_utt, settings_dict):
    if settings_dict["loud"]["is_enabled"]:
        sys_utt = sys_utt.upper()
    if settings_dict["voice_assistant"]["is_enabled"]:
        engine.say(sys_utt)
        engine.runAndWait()

    return sys_utt


# get sys_utt and check if delayed option is set
# returns label and user_utt
def get_user_input(sys_utt, settings_dict):
    # NOTE: this also calls get_sys_utt based on settings
    user_utt = input(get_sys_utt(sys_utt, settings_dict))
    if settings_dict["delayed"]["is_enabled"]:
        time.sleep(4)
    label = classifier(user_utt.lower())
    log.debug(f"Classified utterance as {label}")

    return label, user_utt


def start(list_models, optional):
    global classifier, dialog_tree, current_node, classifier, form

    # global variables
    dialog_tree = None
    classifier = None
    current_node = "welcome"
    form = {"food": "", "pricerange": "", "area": "", "extra_preference": ""}

    dialog_tree = lib.create_dialog_tree()
    settings_dict = lib.create_settings_dict()
    models_dict = lib.create_models_dict(list_models)

    if optional == "":  # show menu normally
        selected_settings = lib.show_options_menu(settings_dict, "Configure your desired settings", True)
        selected_method = lib.show_options_menu(models_dict, "Select your classification model")
        # enable based on menu
        settings_dict = enable_settings(settings_dict, selected_settings)
        classifier = lib.enable_method(models_dict, selected_method)
    else:
        if optional == "A" or optional == "B":
            classifier = lib.enable_method(models_dict, ('', 1))
        if optional == "A":  # delayed
            settings_dict = enable_settings(settings_dict, [('', 2)])
        if optional == "B":  # not delayed
            settings_dict = enable_settings(settings_dict, [])
    log.debug(classifier, settings_dict)

    while current_node != "goodbye":
        mode = dialog_tree[current_node]["mode"]
        sys_utt = dialog_tree[current_node]["sys_utt"]
        exits = dialog_tree[current_node]["exits"]
        conditions = dialog_tree[current_node]["exit_conditions"]

        log.debug(f"\nCurrent Node: {current_node}\nMode: {mode}")
        log.debug(f"\nExits: {exits}\nConditions: {conditions}\nForm: {form}")

        mode_split = mode.split("_", maxsplit=1)

        if mode_split[0] in ["ask", "extract", "welcome"]:
            label, user_utt = get_user_input(sys_utt, settings_dict)
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
            label, user_utt = get_user_input(sys_utt, settings_dict)
            restaurant.set_recommendations(form)

        elif mode == "suggest":
            recommendations = restaurant.get_recommendations()
            index, sys_utt = restaurant.get_recommendations_message()
            sys_utt = f"Found {len(recommendations)} matching restaurant(s).\n" + sys_utt

            if recommendations.empty:
                print(get_sys_utt(sys_utt, settings_dict))
                label = "null"
            else:
                label, user_utt = get_user_input(sys_utt, settings_dict)
                restaurant.drop_recommendation(index)

        # evaluate exit conditions
        for i, condition in enumerate(conditions):
            evaluation = eval(condition)
            log.debug(f"condition {i}: {condition} is {evaluation}")

            if evaluation:
                next_node = exits[i]
                break

        current_node = next_node
    print(get_sys_utt("Goodbye! Thanks for using our bot!", settings_dict))
