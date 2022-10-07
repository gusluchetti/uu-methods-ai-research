import logging
from Levenshtein import distance

log = logging.getLogger(__name__)


# IMPORTANT TODO:
# populate list of price ranges, locations and food types based on
# restaurant dataframe uniques (for each of the columns?)
def best_word_match(word_to_match, wordtype_l):
    """
    takes misspelled word and matches it with nearest
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


def extract_preference(user_utt):
    """
    args : user_utt - user
    return : dictionary of fields corresponding to global preference_form
        filled with a (keyword | "any" | "")
    """
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

    extra_pref_list = [
        "touristic",
        "children",
        "assigned seats",
        "romantic",
        "seats assigned",
    ]

    # preprocessing
    sentence_string = user_utt
    sentence_string = sentence_string.lower()
    sentence_string = sentence_string.replace("north american", "northamerican")
    sentence_string = sentence_string.replace("modern european", "moderneuropean")
    sentence_string = sentence_string.replace("asian oriental", "asianoriental")
    sentence_string = sentence_string.replace("assigned seats", "assignedseats")
    sentence_string = sentence_string.replace("seats assigned", "seatsassigned")
    sentence = list(sentence_string.split(" "))
    sentence = [
        "north american" if item == "northamerican" else item for item in sentence
    ]
    sentence = [
        "modern european" if item == "moderneuropean" else item for item in sentence
    ]
    sentence = [
        "asian oriental" if item == "asianoriental" else item for item in sentence
    ]
    sentence = [
        "seats assigned" if item == "seatsassigned" else item for item in sentence
    ]
    sentence = [
        "assigned seats" if item == "assignedseats" else item for item in sentence
    ]

    type_of_food = ""
    location = ""
    pricerange = ""
    extra_preference = ""

    # find the index of the word on pricerange
    price_ranges = [
        "any",
        "cheap",
        "moderate",
        "expensive"
    ]
    pricerange_index = -1

    for price in price_ranges:
        if price in sentence:
            pricerange_index = sentence.index(price)
            pricerange = price

    # If no keyword was matched, look at positions in sentence where you would
    # expect to find a price. Choose the word from that position
    # which has the lowest Levenshtein edit distance from our keywords
    if pricerange == "":
        pricerange_candidates = []
        for word in sentence:
            if (word == "priced") and (sentence.index(word) != 0):
                pricerange_candidates.append(sentence[sentence.index(word) - 1])
        if best_match_list(pricerange_candidates, ["moderate", "cheap", "expensive"]):
            pricerange, misspelled_pricerange = best_match_list(
                pricerange_candidates, ["moderate", "cheap", "expensive"]
            )
            pricerange_index = sentence.index(misspelled_pricerange)

    log.debug(f"got {pricerange} price range")

    # find the index of the word on type of food.
    # make sure it is NOT the same as the one for pricerange
    type_index = -2
    type_candidates = []
    for i, word in enumerate(sentence):
        if word in food_types and word != "any":
            type_of_food = word
            type_index = sentence.index(word)
            break
        if (
            (word == "food" or word == "restaurant")
            and (sentence.index(word) != 0)
            and (sentence.index(word) - 1 != pricerange_index)
        ):
            type_candidates.append(sentence[sentence.index(word) - 1])
        elif (
            (word == "serving" or word == "serves")
            and (sentence.index(word) != sentence.index(sentence[-1]))
            and (sentence.index(word) + 1 != pricerange_index)
        ):
            type_candidates.append(sentence[sentence.index(word) + 1])
    if best_match_list(type_candidates, food_types):
        type_of_food, misspelled_type = best_match_list(type_candidates, food_types)
        type_index = sentence.index(misspelled_type)

    log.debug(f"got {type_of_food} food type")

    # find the index of the word on location.
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
        if local in sentence:
            location_index = sentence.index(local)
            location = local

    # If no keyword was matched, look at positions in sentence where you would
    # expect to find a location. If no keyword relative to another category (type
    # or pricerange) was found at that position, choose the word from that position
    # which has the lowest Levenshtein edit distance from our keywords
    if location == "":
        location_candidates = []
        for word in sentence:
            if (
                (word == "area" or word == "part")
                and (sentence.index(word) != 0)
                and (sentence.index(word) - 1 != type_index)
                and (sentence.index(word) - 1 != pricerange_index)
            ):
                location_candidates.append(sentence[sentence.index(word) - 1])
        if best_match_list(location_candidates, locations):
            location, misspelled_location = best_match_list(
                location_candidates, locations
            )
            location_index = sentence.index(misspelled_location)

    log.debug(f"got {location} location")

    # find extra preferences, if the user has any
    if best_match_list(sentence, extra_pref_list):
        extra_preference, misspelled_extra_pref = best_match_list(
            sentence, extra_pref_list
        )
        if (
            sentence.index(misspelled_extra_pref) == location_index
            or sentence.index(misspelled_extra_pref) == pricerange_index
            or sentence.index(misspelled_extra_pref) == type_index
        ):
            extra_preference = "unknown"

    return {
        "area": location,
        "pricerange": pricerange,
        "food": type_of_food,
        "extra_preference": extra_preference,
    }


def extract_food(user_utt):
    return extract_preference(user_utt)["food"]


def extract_area(user_utt):
    return extract_preference(user_utt)["area"]


def extract_pricerange(user_utt):
    return extract_preference(user_utt)["pricerange"]


def extract_extra_preference(user_utt):
    return extract_preference(user_utt)["extra_preference"]
