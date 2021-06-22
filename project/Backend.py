import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle


def RandomForestModel(X, y):
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    rf_model = RandomForestClassifier()
    # Fit the model on training set
    rf_model.fit(X_train_tfidf, y_train)

    # save the model to disk
    filename = 'rf_model1.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    y_preds = rf_model.predict(X_test_tfidf)
    acc1 = accuracy_score(y_test, y_preds)
    report = classification_report(y_test, y_preds)
    print(report)
    print("Random Forest, Accuracy Score:", acc1)
    # confusion_matrix_cal(y_test, y_preds)
    return y_test, y_preds


def tfidf_representation(tweets):
    global tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
    tfidf = tfidf_vectorizer.fit_transform(tweets.values.astype('U'))
    print(tfidf.shape)
    return tfidf


def preprocessing(tweet):
    stopwords = nltk.corpus.stopwords.words("english")

    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()

    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ')

    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '')

    giant_url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_name.str.replace(giant_url_regex, '')

    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    newtweet = punc_remove.str.replace(r'\s+', ' ')
    newtweet = newtweet.str.replace(r'^\s+|\s+?$', '')
    newtweet = newtweet.str.replace(r'\d+(\.\d+)?', 'numbr')
    tweet_lower = newtweet.str.lower()

    tokenized_tweet = tweet_lower.apply(lambda x: x.split())

    tokenized_tweet = tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p = tokenized_tweet

    return tweets_p


def confusion_matrix_cal(y_test, y_preds):
    confusion_matrix1 = confusion_matrix(y_test, y_preds)
    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = confusion_matrix1[i, :] / float(confusion_matrix1[i, :].sum())
    names = ['Hate', 'Offensive', 'Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='YlGnBu', cbar=False, square=True, fmt='.2f')
    plt.ylabel(r'True Value', fontsize=14)
    plt.xlabel(r'Predicted Value', fontsize=14)
    plt.tick_params(labelsize=12)


def prediction(test):
    main()
    test = preprocess_test(test)
    print(test)
    test = [test]
    print(test)
    test_tfidf = tfidf_vectorizer.transform(test)
    # load the model from disk
    filename = 'rf_model1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    predicted = loaded_model.predict(test_tfidf)
    predicted = predicted[0]
    if predicted == 0:
        return "Hate Speech"
    elif predicted == 1:
        return "Offensive language"
    else:
        return "Neither"


def preprocess_test(newtweet):
    stemmer = PorterStemmer()
    tweet_lower = newtweet.lower()
    tokenized_tweet = tweet_lower.split()
    tokenized_tweet = list(map(lambda x: stemmer.stem(x), tokenized_tweet))
    return " ".join(tokenized_tweet)


def main():
    df = pd.read_csv(r"C:\Users\Moataz\Desktop\SEMSTER_B\big data\project\DStoolsProject\clean_dataset.csv")
    tfidf = tfidf_representation(df['processed_tweets'])


main()
