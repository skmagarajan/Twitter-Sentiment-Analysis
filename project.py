import numpy as np
import pandas as pd
import re
import time
import ssl
import functools
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk import TweetTokenizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics, linear_model, ensemble
from sklearn.metrics import precision_recall_fscore_support


def filter_obama(data, dropnan=None):
    raw_obama_dataframe = pd.read_excel(data, 'Obama', header=0, usecols="D,E")
    raw_obama_dataframe['Class'] = pd.to_numeric(raw_obama_dataframe['Class'], errors='coerce', downcast='float')
    filtered_obama_df = raw_obama_dataframe[raw_obama_dataframe.Class != 2]
    if dropnan:
        filtered_obama_df = filtered_obama_df.dropna()

    return filtered_obama_df


def filter_romney(data, dropnan=None):
    raw_romney_dataframe = pd.read_excel(data, 'Romney', header=0, usecols="D,E")
    raw_romney_dataframe['Class'] = pd.to_numeric(raw_romney_dataframe['Class'], errors='coerce', downcast='float')
    filtered_romney_df = raw_romney_dataframe[raw_romney_dataframe.Class != 2]

    if dropnan:
        filtered_romney_df = filtered_romney_df.dropna()

    return filtered_romney_df


def obama_testing(data):
    obama_test = pd.read_excel(data, "Obama", header=None, usecols="A,B")
    obama_test.rename(columns={0: 'id', 1: 'Annotated Tweet'}, inplace=True)
    return obama_test


def romney_testing(data):
    romney_test = pd.read_excel(data, "Romney", header=None, usecols="A,B")
    romney_test.rename(columns={0: 'id', 1: 'Annotated Tweet'}, inplace=True)
    return romney_test


# Preprocessing
def PreProcessing(data, dictionary):
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('[^\x00-\x7f]', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].apply(lambda s: " ".join(
        [dictionary[word.lower()] if word.lower() in dictionary.keys() else word for word in s.split()]))
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',
                                                                  ' ')
    special_stopwords = ['omg', 'lol', 'umm', 'hmm', 'ah', 'oh', 'yea']
    special_stopwords.extend(stopwords.words('english'))
    data["Annotated Tweet"] = data["Annotated Tweet"].apply(
        lambda s: " ".join([word for word in s.split() if word not in special_stopwords]))
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('<[^<]+?>', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('^(!|\.|,|<|>|:|;|{|}|\||~)$', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('\d+', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('\n\n', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('\n', ' ')
    data['Annotated Tweet'] = data['Annotated Tweet'].str.replace('[^\w\s]', ' ')
    data["Annotated Tweet"] = data["Annotated Tweet"].str.replace('\s+', ' ')
    data["Annotated Tweet"] = data["Annotated Tweet"].apply(
        lambda s: re.compile(r"(.)\1{1,}", re.DOTALL).sub(r"\1\1", s))
    data["Annotated Tweet"] = data["Annotated Tweet"].apply(lambda s: re.sub("([a-z])([A-Z])", "\g<1> \g<2>", s))
    data["Annotated Tweet"] = data["Annotated Tweet"].apply(lambda s: re.sub(r"^\s+", "", s))
    return data


def negative_word_conversion():
    dictionary = {}
    f = open("/Users/ashwin/Desktop/change.txt")
    lines = f.readlines()
    f.close()
    for i in lines:
        try:
            tmp = i.replace('"', '').replace(',', '').replace('\n', ' ').split(':')
            dictionary[tmp[0]] = tmp[1]
        except:
            print(tmp)
            print(z)
    return dictionary


def model(model):
    clf = False
    if model == "svm":
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LinearSVC(class_weight="balanced"))])
    elif model == "bayes":
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])
    elif model == "knn":
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', KNeighborsClassifier())])
    elif model == "sgd":
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', linear_model.SGDClassifier(class_weight="balanced"))])
    elif model == "log_reg":
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                        ('tfidf', TfidfTransformer()),
                        ('clf', linear_model.LogisticRegression(class_weight="balanced"))])

    return clf


def score_reducer(accum, new_score, divisor):
    return {"Precision": new_score["Precision"] / divisor + accum["Precision"],
            "Recall": new_score["Recall"] / divisor + accum["Recall"],
            "Fscore": new_score["Fscore"] / divisor + accum["Fscore"]}


def KFoldCrossValidation(model, tweet, type, fold, stratified=None):
    positive_score = []
    normal_score = []
    negative_score = []
    accuracy = []
    random_index = np.random.permutation(len(tweet))
    bucket_size = int(len(tweet) / fold)
    balance = len(tweet) % fold

    if not stratified:
        for x in range(1, fold + 1):
            test = random_index[(x - 1) * bucket_size:x * bucket_size]
            train = np.concatenate([random_index[:(x - 1) * bucket_size], random_index[x * bucket_size:]])
            model.fit(tweet.iloc[train], type.iloc[train])
            predict = model.predict(tweet.iloc[test])
            precision, recall, fscore, support = precision_recall_fscore_support(type.iloc[test], predict,
                                                                                 labels=[1, 0, -1])
            positive_score.append({"Precision": precision[0], "Recall": recall[0], "Fscore": fscore[0]})
            normal_score.append({"Precision": precision[1], "Recall": recall[1], "Fscore": fscore[1]})
            negative_score.append({"Precision": precision[2], "Recall": recall[2], "Fscore": fscore[2]})
            accuracy.extend(predict == type.iloc[test])
    else:
        stratify = StratifiedKFold(n_splits=fold, shuffle=True)
        for train, test in stratify.split(tweet, type):
            model.fit(tweet.iloc[train], type.iloc[train])
            predict = model.predict(tweet.iloc[test])
            precision, recall, fscore, support = precision_recall_fscore_support(type.iloc[test], predict,
                                                                                 labels=[1, 0, -1])
            positive_score.append({"Precision": precision[0], "Recall": recall[0], "Fscore": fscore[0]})
            normal_score.append({"Precision": precision[1], "Recall": recall[1], "Fscore": fscore[1]})
            negative_score.append({"Precision": precision[2], "Recall": recall[2], "Fscore": fscore[2]})
            accuracy.extend(predict == type.iloc[test])

    positive = functools.reduce(functools.partial(score_reducer, divisor=fold), positive_score,
                                {"Precision": 0, "Recall": 0, "Fscore": 0})
    neutral = functools.reduce(functools.partial(score_reducer, divisor=fold), normal_score,
                               {"Precision": 0, "Recall": 0, "Fscore": 0})
    negative = functools.reduce(functools.partial(score_reducer, divisor=fold), negative_score,
                                {"Precision": 0, "Recall": 0, "Fscore": 0})
    accuracy = np.mean(accuracy)

    print("Positive: ", positive)
    print("Neutral: ", neutral)
    print("Negative: ", negative)
    print("Accuracy: %f" % accuracy)


def print_class(test, train, output, val):
    print_result = {'id': test['id'], 'Class': train}
    print_df = pd.DataFrame(print_result)
    np.savetxt(output, print_df, fmt=['%d', '%d'], delimiter=';;', header=val, comments='')


if __name__ == "__main__":
    # Obama Data
    obama_filtered = filter_obama(r"/Users/ashwin/Desktop/training-Obama-Romney-tweets.xlsx", dropnan=True)
    # Romney Data
    romney_filtered = filter_romney(r"/Users/ashwin/Desktop/training-Obama-Romney-tweets.xlsx", dropnan=True)

    # Pre Processing of both Obama and Romney datasets
    obama_filtered = PreProcessing(obama_filtered, negative_word_conversion())
    romney_filtered = PreProcessing(romney_filtered, negative_word_conversion())

    # Model 1: SVM
    obama_dataframe = model("svm")
    romney_dataframe = model("svm")
    print("--------Model 1 - Support Vector Machine----------")
    print("OBAMA:")
    KFoldCrossValidation(obama_dataframe, obama_filtered['Annotated Tweet'], obama_filtered['Class'], 10,
                         stratified=True)
    print("\nROMNEY:")
    KFoldCrossValidation(romney_dataframe, romney_filtered['Annotated Tweet'], romney_filtered['Class'], 10,
                         stratified=True)

    # Model 2: Naive Bayes
    obama_dataframe = model("bayes")
    romney_dataframe = model("bayes")
    print("\n--------Model 2 - Multinominal Naive Bayes----------")
    print("OBAMA:")
    KFoldCrossValidation(obama_dataframe, obama_filtered['Annotated Tweet'], obama_filtered['Class'], 10,
                         stratified=True)
    print("\nROMNEY:")
    KFoldCrossValidation(romney_dataframe, romney_filtered['Annotated Tweet'], romney_filtered['Class'], 10,
                         stratified=True)

    # Model 3: K-Nearest Neighbour
    obama_dataframe = model("knn")
    romney_dataframe = model("knn")
    print("\n--------Model 3 - K Nearest Neighbour----------")
    print("OBAMA:")
    KFoldCrossValidation(obama_dataframe, obama_filtered['Annotated Tweet'], obama_filtered['Class'], 10,
                         stratified=True)
    print("\nROMNEY:")
    KFoldCrossValidation(romney_dataframe, romney_filtered['Annotated Tweet'], romney_filtered['Class'], 10,
                         stratified=True)

    # Model 4: SGD
    obama_dataframe = model("sgd")
    romney_dataframe = model("sgd")
    print("\n--------Model 4 - Stochastic Gradient Descent----------")
    print("OBAMA:")
    KFoldCrossValidation(obama_dataframe, obama_filtered['Annotated Tweet'], obama_filtered['Class'], 10,
                         stratified=True)
    print("\nROMNEY:")
    KFoldCrossValidation(romney_dataframe, romney_filtered['Annotated Tweet'], romney_filtered['Class'], 10,
                         stratified=True)

    # Model 5: Logistic Regression
    obama_dataframe = model("log_reg")
    romney_dataframe = model("sgd")
    print("\n--------Model 5 - Logistic Regression----------")
    print("OBAMA:")
    KFoldCrossValidation(obama_dataframe, obama_filtered['Annotated Tweet'], obama_filtered['Class'], 10,
                         stratified=True)
    print("\nROMNEY:")
    KFoldCrossValidation(romney_dataframe, romney_filtered['Annotated Tweet'], romney_filtered['Class'], 10,
                         stratified=True)
