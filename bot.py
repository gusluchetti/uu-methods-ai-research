def classifier(ut):
  return ut

def extract_location(ut):
  return ut
def extract_cuisine(ut):
  return ut
def extract_price(ut):
  return ut
def suggest_restaurants():
  sug = ['chinese, NY, cheap', 'italian, Napoli, expensive', 'greek, Athens, cheap']
  return sug

# passing functions that return predictions for the bot to use
def bot(list_models):
    print("Hello! I'm a restaurant recommendation bot! \n")
    classifier_key = input("""
Please select a classification method (first two are baseline systems):
[1] - Majority Class
[2 (Default)] Keyword Matching
[3] - Logistic Regression
[4] - Multinomial Naive-Bayes\n""")

    # set desired classifier
    classifier = list_models["2"]
    try:
        if classifier_key in list_models.keys():
            classifier = list_models[classifier_key]
            print(f"Model {classifier} was selected!")
        else:
            print("Model doesn't exist!")
    except:
        print('whoops! - bad things happened somehow')

    # starting bot up
 #   finished = False
 #   while not finished:
 #       utterance = input('>').lower()
 #       label = classifier(utterance)
 #       print(f"Your last input was labeled/classified as: {label}")
 #       if utterance == "forcequit" or label == "bye":
 #           finished = True
 #           print("Bye! See ya later.")

    finished = False
    while not finished:
      label = 'null'
      while label != 'inform':
        print('Hi! What would you like to eat?')
        utterance = input('>').lower()
        label = classifier(utterance)
        if label == 'inform':
          location = extract_location(utterance)
          cuisine_type = extract_cuisine(utterance)
          price_range = extract_price(utterance)
        else:
          continue
        while not location:
          print("Sorry I didn't get the location. Where would you like to eat?")
          utterance = input('>').lower()
          location = extract_location(utterance)
        print('I got the location!')
        while not cuisine_type:
          print("Sorry I didn't get the type of cuisine. What would you like to eat?")
          utterance = input('>').lower()
          cuisine_type = extract_cuisine(utterance)
        print('I got the cusine type!')
        while not price_range:
          print("Sorry I didn't get the pricing of the food. Expensive? Cheap? Moderate?")
          utterance = input('>').lower()
          price_range = extract_price(utterance)
        print('I got the price range!')

        while label not in ['negate','deny','confirm','affirm','thankyou','ack']:
          print('Is this what you want?')
          print(f'location: {location}\ncuisine type:{cuisine_type}\nprice range:{price_range}')
          utterance = input('>').lower()
          label = classifier(utterance)
        
        suggestions = None
        if label in ['confirm','affirm', 'thankyou', 'ack']:
          suggestions = suggest_restaurants()
          label = 'null'
        elif label in ['negate','deny']:
          label = 'null'
          continue

        while label not in ['negate','deny','confirm','affirm','thankyou','ack']:
          print('Do you like suggestion?')
          print(suggestions[0])
          utterance = input('>').lower()
          label = classifier(utterance)
        
        if label in ['confirm','affirm', 'thankyou', 'ack']:
          finished = True
          print('Bye!')
          break
        elif label in ['negate','deny']:
          for sug in suggestions[1:]:
            print('Do you like this suggestion?')
            print(sug)
            utterance = input('>').lower()
            label = classifier(utterance)
            if label in ['confirm','affirm','thankyou','ack']:
              finished = True
              label = 'inform'
              print('Bye!')
              break
          if label != 'inform':
            print('Sorry we are out of options matching your criteria. Lets try again!')
            
