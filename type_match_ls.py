from Levenshtein import distance


def bestMatchWord(word_to_match, wordtype_l):
    '''takes misspelled word and matches it with nearest value of the list of words of the same type.
    The used distance in the levenshtein edit distance.
    
    args: word_to_match (str) -> the misspelled word
          wordtype_l (list) -> list of words of the same type as word_to_match'''
    
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


def bestMatchList(list_of_words, wordtype_list):
    'same as bestMatch but with lists as input'
    bestdist = 4
    bestmatch = ""
    misspelled = ""
    dist = ""
    match = "" 

    for word in list_of_words:
        if bestMatchWord(word, wordtype_list):
            match, dist = bestMatchWord(word, wordtype_list)
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
    args :
      user_utt - user utterance, example "I'm looking for a moderately priced restaurant in the west part of town"
    return :
      dictionary of fields corresponding to global preference_form filled with a (keyword | "unknown" | "any")
    """
    # 'WE ADDED SWEDISH AND WORLD IN THE TYPES OF FOOD
    # SINCE THEY WERE IN THE EXAMPLES FOR THE TASK, EVEN THOUGH THEY ARE NOT IN OUR DATABASE'
    food_types = ["african", "any", "asian oriental", "australasian", "bistro", "british", "catalan", "chinese", "cuban",
                          "european", "french", "fusion","gastropub", "indian","international", "italian", "jamaican","japanese",
                          "korean","lebanese","mediterranean","modern european","moroccan", "north american", "persian","polynesian","portuguese","romanian",
                          "seafood", "spanish","steakhouse","swiss","thai","traditional","turkish","tuscan","vietnamese","world","swedish"]  
    sentence_string = user_utt
    # preprocessing
    sentence_string = sentence_string.lower()
    sentence_string = sentence_string.replace("north american", "northamerican")
    sentence_string = sentence_string.replace("modern european", "moderneuropean")
    sentence_string = sentence_string.replace("asian oriental", "asianoriental")
    sentence = list(sentence_string.split(" "))
    sentence = ["north american" if item == "northamerican" else item for item in sentence]
    sentence = ["modern european" if item == "moderneuropean" else item for item in sentence]
    sentence = ["asian oriental" if item == "asianoriental" else item for item in sentence]

    type_of_food = "unknown"
    location = "unknown"
    pricerange = "unknown"

    'Find the index of the word on pricerange.'
    pricerange_index = -1
    if "cheap" in sentence:
        pricerange_index = sentence.index("cheap")
        pricerange = "cheap"
    elif "expensive" in sentence:
        pricerange_index = sentence.index("expensive")
        pricerange = "expensive"
    elif "moderate" in sentence:
        pricerange_index = sentence.index("moderate")
        pricerange = "moderate"
    elif ("any cost" in sentence_string) or ("any price" in sentence_string):
        i = 0
        end = False
        while not end:
            if (sentence[i] == "any") and ((sentence[i + 1] == "price") or (sentence[i + 1] == "cost")):
                pricerange_index = i
                pricerange = sentence[i]
                end = True
            i = i + 1
    else:
        '''If no keyword was matched, look at positions in sentence where you would
        expect to find a price. Choose the word from that position
        which has the lowest Levenshtein edit distance from our keywords.'''
        pricerange_candidates = []
        for word in sentence:
            if (word == "priced") and (sentence.index(word) != 0):
                pricerange_candidates.append(sentence[sentence.index(word) - 1])
        if bestMatchList(pricerange_candidates, ["moderate", "cheap", "expensive"]):
            pricerange, misspelled_pricerange = bestMatchList(pricerange_candidates, ["moderate", "cheap", "expensive"])
            pricerange_index = sentence.index(misspelled_pricerange)

            
    ''''Find the index of the word on type of food. Make sure it is not the same as the
    one for pricerange'''
    type_index = -2
    type_candidates = []
    for i, word in enumerate(sentence):
        if word in food_types and word != "any":
            type_of_food = word
            type_index = sentence.index(word)
            break
        if (word == "food" or word == "restaurant") and (sentence.index(word) != 0) and (sentence.index(word) - 1 != pricerange_index):
            type_candidates.append(sentence[sentence.index(word) - 1])
        elif (word == "serving" or word == "serves") and (sentence.index(word) != sentence.index(sentence[-1])) and (sentence.index(word) + 1 != pricerange_index):
            type_candidates.append(sentence[sentence.index(word) + 1])
    if bestMatchList(type_candidates, food_types):
        type_of_food, misspelled_type = bestMatchList(type_candidates, food_types)
        type_index = sentence.index(misspelled_type)

    'Find the index of the word on location.'
    location_index = -3
    if "north" in sentence:
        location_index = sentence.index("north")
        location = "north"
    elif "south" in sentence:
        location_index = sentence.index("south")
        location = "south"
    elif "west" in sentence:
        location_index = sentence.index("west")
        location = "west"
    elif "east" in sentence:
        location_index = sentence.index("east")
        location = "east"
    elif "centre" in sentence:
        location_index = sentence.index("centre")
        location = "centre"
    elif "center" in sentence:
        location_index = sentence.index("center")
        location = "centre"
    else:
        '''If no keyword was matched, look at positions in sentence where you would
        expect to find a location. If no keyword relative to another category (type
        or pricerange) was found at that position, choose the word from that position
        which has the lowest Levenshtein edit distance from our keywords.'''
        location_candidates = []
        for word in sentence:
            if (word == "area" or word == "part") and (sentence.index(word) != 0) and (sentence.index(word) - 1 != type_index) and (sentence.index(word) - 1 != pricerange_index):
                location_candidates.append(sentence[sentence.index(word) - 1])
        if bestMatchList(location_candidates, ["any", "north", "south", "west", "east", "centre"]):
            location, misspelled_location = bestMatchList(location_candidates, ["any","north", "south", "west", "east", "centre"])
            location_index = sentence.index(misspelled_location)
    
    return {'area': location,
            'pricerange': pricerange,
            'food': type_of_food
            }

def extract_food(user_utt):
    return extract_preference(user_utt)['food']

def extract_area(user_utt):
    return extract_preference(user_utt)['area']

def extract_pricerange(user_utt):
    return extract_preference(user_utt)['pricerange']