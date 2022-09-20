# passing functions that return predictions for the bot to use
def bot(list_models):
    print("Hello! I'm a restaurant recommendation bot! \n")
    classifier_key = input("""
    Please select a classifier method:
    (1-Default) - 1st Baseline - Majority Class: Always predicts the majority class (in this case, 'inform' label)
    (2) - 2nd Baseline - Keyword Matching: Predictions are based on keywords found in the utterance
    (3) - Multinomial Naive-Bayes
    (4) - Logistic Regression
    """)

    # set desired classifier
    classifier = list_models["1"]
    try:
        if classifier_key in list_models.keys():
            classifier = list_models[classifier_key]
            print(f"Model {classifier} was selected!")
        else:
            print('Model doesnt exist!')
    except:
        print('whoops! - bad things happened somehow')

    # starting bot up
    finished = False
    while not finished:
        utterance = input('>').lower()
        classification = classifier(utterance)
        print(f"Your last input was labeled/classified as : {classification}")

        if utterance == "forcequit" or classification == "bye":
            finished = True
            print("Bye! See ya later.")
