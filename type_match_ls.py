import logging
from Levenshtein import distance

log = logging.getLogger(__name__)


def best_word_match(word_to_match, wordtype_l):
    """
    Takes misspelled word and matches it with nearest
    value of the list of words of the same type.

    args: word_to_match (str) -> the misspelled word
          wordtype_l (list) -> list of words of the same type as word_to_match
    """

    mindist = 4
    match = ""

    for value in wordtype_l:
        dist = distance(value, word_to_match)
        if dist < mindist:
            mindist = dist
            match = value

    if mindist <= 3:
        return match, mindist
    else:
        return False


def best_match_list(list_of_words, wordtype_list):
    "same as bestMatch but with lists as input"
    bestdist = 4
    bestmatch = ""
    misspelled = ""
    dist = ""
    match = ""

    for word in list_of_words:
        if best_word_match(word, wordtype_list):
            match, dist = best_word_match(word, wordtype_list)
            if dist <= bestdist:
                bestdist = dist
                bestmatch = match
                misspelled = word

    if bestdist == 4:
        return False
    else:
        return bestmatch, misspelled


def pre_process(user_utt):
    return list(user_utt.lower().split(" "))


form = {
    "area": "",
    "pricerange": "",
    "food": "",
    "extra_preference": ""
}
location_index = -1
pricerange_index = -1
type_index = -1


def extract_area(user_utt):
    user_utt = pre_process(user_utt)
    global form, location_index
    location = ""
    locations = [
        "any",
        "north",
        "east",
        "west",
        "south",
        "center"
    ]
    location_index = -3
    for local in locations:
        if local in user_utt:
            location_index = user_utt.index(local)
            location = local

    # If no keyword was matched, look at positions in user_utt where you would
    # expect to find a location. If no keyword relative to another category (type
    # or pricerange) was found at that position, choose the word from that position
    # which has the lowest Levenshtein edit distance from our keywords
    if location == "":
        location_candidates = []
        for word in user_utt:
            if (
                (word == "area" or word == "part")
                and (user_utt.index(word) != 0)
                and (user_utt.index(word) - 1 != type_index)
                and (user_utt.index(word) - 1 != pricerange_index)
            ):
                location_candidates.append(user_utt[user_utt.index(word) - 1])
        if best_match_list(location_candidates, locations):
            location, misspelled_location = best_match_list(
                location_candidates, locations
            )
            location_index = user_utt.index(misspelled_location)

    log.debug(f"got {location} location")

    form["area"] = location
    return location


def extract_pricerange(user_utt):
    user_utt = pre_process(user_utt)
    global form, pricerange_index
    pricerange = ""
    # find the index of the word on pricerange
    price_ranges = [
        "any",
        "cheap",
        "moderate",
        "expensive"
    ]
    pricerange_index = -1

    for price in price_ranges:
        if price in user_utt:
            pricerange_index = user_utt.index(price)
            pricerange = price

    # If no keyword was matched, look at positions in sentence where you would
    # expect to find a price. Choose the word from that position
    # which has the lowest Levenshtein edit distance from our keywords
    if pricerange == "":
        pricerange_candidates = []
        for word in user_utt:
            if (word == "priced") and (user_utt.index(word) != 0):
                pricerange_candidates.append(user_utt[user_utt.index(word) - 1])
        if best_match_list(pricerange_candidates, ["moderate", "cheap", "expensive"]):
            pricerange, misspelled_pricerange = best_match_list(
                pricerange_candidates, ["moderate", "cheap", "expensive"]
            )
            pricerange_index = user_utt.index(misspelled_pricerange)

    log.debug(f"got {pricerange} price range")

    form["pricerange"] = pricerange
    return pricerange


def extract_food(user_utt):
    user_utt = pre_process(user_utt)
    global form, type_index
    food_types = [
        "african",
        "any",
        "asian oriental",
        "australasian",
        "bistro",
        "british",
        "catalan",
        "chinese",
        "cuban",
        "european",
        "french",
        "fusion",
        "gastropub",
        "indian",
        "international",
        "italian",
        "jamaican",
        "japanese",
        "korean",
        "lebanese",
        "mediterranean",
        "modern european",
        "moroccan",
        "north american",
        "persian",
        "polynesian",
        "portuguese",
        "romanian",
        "seafood",
        "spanish",
        "steakhouse",
        "swiss",
        "thai",
        "traditional",
        "turkish",
        "tuscan",
        "vietnamese",
    ]

    type_of_food = ""
    # find the index of the word on type of food.
    # make sure it is NOT the same as the one for pricerange
    type_index = -2
    type_candidates = []
    for i, word in enumerate(user_utt):
        if word in food_types:
            type_of_food = word
            type_index = user_utt.index(word)
            break
        if (
            (word == "food" or word == "restaurant")
            and (user_utt.index(word) != 0)
            and (user_utt.index(word) - 1 != pricerange_index)
        ):
            type_candidates.append(user_utt[user_utt.index(word) - 1])
        elif (
            (word == "serving" or word == "serves")
            and (user_utt.index(word) != user_utt.index(user_utt[-1]))
            and (user_utt.index(word) + 1 != pricerange_index)
        ):
            type_candidates.append(user_utt[user_utt.index(word) + 1])
    if best_match_list(type_candidates, food_types):
        type_of_food, misspelled_type = best_match_list(type_candidates, food_types)
        type_index = user_utt.index(misspelled_type)

    log.debug(f"got {type_of_food} food type")

    form["food"] = type_of_food
    return type_of_food


def extract_extra_preference(user_utt):
    user_utt = pre_process(user_utt)
    global form, location_index, pricerange_index, type_index
    extra_preference = ""
    extra_pref_list = [
        "touristic",
        "children",
        "assigned seats",
        "romantic",
        "seats assigned",
    ]
    # find extra preferences, if the user has any
    if best_match_list(user_utt, extra_pref_list):
        extra_preference, misspelled_extra_pref = best_match_list(
            user_utt, extra_pref_list
        )
        if (
            user_utt.index(misspelled_extra_pref) == location_index
            or user_utt.index(misspelled_extra_pref) == pricerange_index
            or user_utt.index(misspelled_extra_pref) == type_index
        ):
            extra_preference = "unknown"

    form["extra_preference"] = extra_preference
    return extra_preference
