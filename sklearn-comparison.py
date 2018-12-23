
import pickle
import pandas as pd
import numpy as np

import spacy

# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.feature_selection import SelectPercentile, chi2

nlp = spacy.load('en')

def extract_words(sentence):
    ignore_words = ['a']
    #words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    doc = nlp(sentence)
    words = [token.lemma_ for token in doc]
    words_cleaned = [w.lower().strip() for w in words if w not in ignore_words]
    return words_cleaned

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)

def extract_entities(sentence):
    doc = nlp(sentence)
    entities = [token.label_ for token in doc.ents]
    return entities

def bagofentities(sentence, ner_tagset):
    entities = extract_entities(sentence)
    # frequency word count
    bag = np.zeros(len(ner_tagset))
    for ent in entities:
        for i, entity in enumerate(ner_tagset):
            if ent == entity:
                bag[i] += 1
    return np.array(bag)

if __name__ == '__main__':

    ner_tagset = ["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART",
                  "LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"]

    df = pd.read_excel('Data/20181001-newindianexpress_sentence_classification_adjudicated_20181218.xlsx')

    cleandf = df[np.logical_not(np.isnan(np.array(df['label'])))]

    cleandf = cleandf.reset_index(drop=True)

    del df

    # vectorizer = TfidfVectorizer()
    # tfidf_vectors = vectorizer.fit_transform(df['sentence'])

    vocab = pd.read_pickle("./vocabulary.pickle")

    feature_list = []
    for i, row in cleandf.iterrows():
        feature_list.append(np.concatenate((bagofwords(row['sentence'], vocab), bagofentities(row['sentence'], ner_tagset))))

    with open('feature_list.pickle', 'wb') as handle:
        pickle.dump(feature_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Created feature_list')

    # feature_list = pd.read_pickle("./feature_list.pickle")

    # All classifiers
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    X = feature_list
    y = np.array(cleandf['label'])

    del cleandf
    del vocab
    del feature_list
    del ner_tagset

    result_file = open("scikit_learn_results.txt", "w")

    for name, classifier in zip(names, classifiers):
        print('Running: ', name)
        y_true, y_pred = y, cross_val_predict(classifier, X, y, n_jobs=5, cv=3)
        result_file.write('\n\n--- ' + name + ' ---')
        result_file.write('\nPrecision:' + str(precision_score(y_true, y_pred, average='weighted')))
        result_file.write('\nRecall:' + str(recall_score(y_true, y_pred, average='weighted')))
        result_file.write('\nF1-score:' + str(f1_score(y_true, y_pred, average='weighted')))
        result_file.write('\nAccuracy:' + str(accuracy_score(y_true, y_pred)))

    result_file.close()
