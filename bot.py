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
    finished = False
    while not finished:
        utterance = input('>').lower()
        label = classifier(utterance)
        print(f"Your last input was labeled/classified as: {label}")
        if utterance == "forcequit" or label == "bye":
            finished = True
            print("Bye! See ya later.")
