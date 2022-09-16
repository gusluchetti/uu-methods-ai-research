from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# bot setup
def bot(classifier):
    list_classifiers = {
        1: "majority class baseline",
        2: "keyword based baseline",
        3: "multinominal nb",
        4: "logistic regression"
    }

    while not finished:
        classifier_key = input("""
            Select your classifier:
            (1) - 1st Baseline - Majority Class: Always predicts the majority class (in this case, 'inform' label)
            (2) - 2nd Baseline - Keyword Matching: Predictions are based on keywords found in the utterance
            (3) - Multinomial Naive-Bayes
            (4) - Logistic Regression
        """)

        try:
            if classifier_key in list_classifiers:
                classifier = list_classifiers[classifier_key]
            else:
                print('Model doesnt exist!')
        except:
            print('whoops!')

        text = input('>').lower()

