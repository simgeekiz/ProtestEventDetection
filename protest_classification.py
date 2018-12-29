#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# ## Load Data

# In[55]:


df = pd.read_excel('Data/20181001-newindianexpress_sentence_classification_adjudicated_20181218.xlsx')


# In[56]:


df = df[np.logical_not(np.isnan(np.array(df['label'])))]


# In[57]:


y = np.array(df['label'])


# # Feature Extraction

# In[58]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[59]:


vectorizer = TfidfVectorizer(min_df=0.002, max_df=0.95, stop_words='english')
tfidf_vectors = vectorizer.fit_transform(df['sentence'])


# In[60]:


# tfidf_vectors


# ### Feature Selection on TF-IDF Vectors

# In[61]:


from sklearn.feature_selection import SelectPercentile, chi2


# In[62]:


tfidf_vectors = SelectPercentile(chi2, percentile=80).fit_transform(tfidf_vectors, y)


# In[63]:


# tfidf_vectors.shape


# ### Named Entity Features

# In[64]:


import spacy


# In[65]:


nlp = spacy.load('en')


# In[66]:


ner_tagset = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
              'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT',
              'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

pos_tagset = ['-LRB-', '-LRB-', ',', ':', '\'\'', '""', '#', '``', '$',
              'ADD', 'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW',
              'HVS', 'HYPH', 'IN',
              'JJ', 'JJR', 'JJS', 'LS', 'MD',
              'NFP', 'NIL', 'NN', 'NNS', 'NNP', 'NNPS',
              'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', '_SP',
              'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
              'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX']


# In[67]:


def bag_of_tags(sentence, ner_tagset, pos_tagset):
    enriched_sentence = nlp(sentence)
    pos_tags = [token.tag_ for token in enriched_sentence if not token.is_stop]
    entities = [token.label_ for token in enriched_sentence.ents]


    # frequency word count
    ner_bag = np.zeros(len(ner_tagset))
    for ent in entities:
        for i, entity in enumerate(ner_tagset):
            if ent==entity:
                ner_bag[i] += 1

    pos_bag = np.zeros(len(pos_tagset))
    for pos in pos_tags:
        for i, postag in enumerate(pos_tagset):
            if pos==postag:
                pos_bag[i] += 1

    return np.concatenate((np.array(ner_bag), np.array(pos_bag)))


# In[68]:


tag_features = []
for i,row in df.iterrows():
    tag_features.append(bag_of_tags(row['sentence'], ner_tagset, pos_tagset))


# ### Combining TF-IDF Vectors and Named Entity Features

# In[71]:


from scipy.sparse import hstack


# In[72]:


X = hstack((tfidf_vectors, np.array(tag_features)))


# #### Saving Feature Vectors

# In[73]:


import pickle


# In[74]:


feature_path = 'Data/feature_list_optimized_Tf_idf_pos_ner_features_sparse_matrix_SCRIPT.pickle'


with open(feature_path, 'wb') as file_:
    pickle.dump(X, file_, protocol=pickle.HIGHEST_PROTOCOL)

# In[ ]:


X = pd.read_pickle(feature_path)


# In[21]:


# type(X)


# #### Memory Cleaning

# In[76]:


del tfidf_vectors
del df
del tag_features
del nlp


# # Classifier Training
# - With hyper-parameter optimization

# https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

# In[88]:


from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report


# In[97]:


opt_results = {}
opt_results_path = 'Results/optimization_results_tfidf_ner_pos_SCRIPT.pickle'


# In[ ]:


scoring = 'f1_macro'


# #### Important Note on the Scoring of Parameter Optimization
# GridSearchCV uses mean accuracy by default. However, we have chosen "f1_macro" scoring for hyper-parameter optimization, because mean accuracy or f1_micro is measuring the performance on the total labels, disregarding the type of the label. So, when we use accuracy or f1_micro, we get high scores because most of the labels are 0 and classifier predicts most of the labels as 0. When the accuracy score for the label 0 is high, the overall result becomes high as well, eventhough the other labels perform low. And this kind of high score doesn't mean our classifier performs better because we are actually interested in getting high scores on label 1 and 2.
#

# ### Decision Tree

# In[80]:


from sklearn.tree import DecisionTreeClassifier


# In[81]:


# Decision Tree
params = {
    'max_depth': [None] + [*range(15, 35, 5)],
    'min_samples_split': [*range(50, 200, 40)],
    'min_samples_leaf': [*range(5, 14, 2)],
    'max_features': [None, 'sqrt', 'log2']
}

dt = DecisionTreeClassifier(criterion='gini')
dt_clf = GridSearchCV(dt, params, cv=5, scoring=scoring)
dt_clf = dt_clf.fit(X, y)


# In[82]:


print('Best Estimator')
print(dt_clf.best_estimator_)
print('Best Score')
print(dt_clf.best_score_)
print('Best Params')
print(dt_clf.best_params_)


# In[92]:


y_true, y_pred = y, cross_val_predict(dt_clf.best_estimator_, X, y, n_jobs=5, cv=5)


# In[93]:


opt_results['DecisionTree'] = {}
opt_results['DecisionTree']['GridSearchCV'] = dt_clf
opt_results['DecisionTree']['classif_report'] = classification_report(y_true, y_pred)


# In[ ]:


print(classification_report(y_true, y_pred))


# In[98]:


with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)


# 0.8481468154012235
# {'max_depth': 20, 'min_samples_split': 100, 'max_features': None, 'min_samples_leaf': 9}
# 699
# <function _passthrough_scorer at 0x2ba712f58950>
# 5
# 0.30017995834350586

# ### RandomForestClassifier

# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


params = {
    'n_estimators': [30, 70, 100, 150],
    'max_depth': [None] + [*range(65, 120, 15)],
    'min_samples_split': [25, 30, 40, 45, 50, 100],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(criterion='gini')
rf_clf = GridSearchCV(rf, params, cv=5, scoring=scoring)
rf_clf = rf_clf.fit(X, y)


# Best Estimator
# RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#             max_depth=80, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=40,
#             min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
# Best Score
# 0.860501379393067
# Best Params
# {'bootstrap': False, 'max_features': 'auto', 'max_depth': 80, 'n_estimators': 30, 'min_samples_split': 40, 'min_samples_leaf': 1}

# In[ ]:


print('Best Estimator')
print(rf_clf.best_estimator_)
print('Best Score')
print(rf_clf.best_score_)
print('Best Params')
print(rf_clf.best_params_)


# In[ ]:


y_true, y_pred = y, cross_val_predict(rf_clf.best_estimator_, X, y, n_jobs=5, cv=5)


# In[ ]:


opt_results['RandomForest'] = {}
opt_results['RandomForest']['GridSearchCV'] = rf_clf
opt_results['RandomForest']['classif_report'] = classification_report(y_true, y_pred)


# In[ ]:


print(classification_report(y_true, y_pred))


# In[ ]:


with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)


# ###  SVC

# In[94]:


from sklearn.svm import SVC


# In[96]:


params = {
    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    'C': [0.025, 0.25, 0.5, 1, 2, 3],
    'gamma': ['auto', 2, 3]
}

svc = SVC()
svc_clf = GridSearchCV(svc, params, cv=5, scoring=scoring)
svc_clf = svc_clf.fit(X, y)


# In[ ]:


print('Best Estimator')
print(svc_clf.best_estimator_)
print('Best Score')
print(svc_clf.best_score_)
print('Best Params')
print(svc_clf.best_params_)


# In[ ]:


y_true, y_pred = y, cross_val_predict(svc_clf.best_estimator_, X, y, n_jobs=5, cv=5)


# In[ ]:


opt_results['SVC'] = {}
opt_results['SVC']['GridSearchCV'] = svc_clf
opt_results['SVC']['classif_report'] = classification_report(y_true, y_pred)


# In[ ]:


print(classification_report(y_true, y_pred))


# In[ ]:


with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)


# 0.8642197433129423
# {'C': 2, 'gamma': 'auto', 'kernel': 'linear'} 'C':[0.025, 0.25, 0.5, 1, 2, 3, 5, 8, 10, 15, 20],
# 48
# <function _passthrough_scorer at 0x2ba712f58950>
# 5
# 9.254388332366943

# ### KNeighborsClassifier

# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[34]:


# p: Power parameter for the Minkowski metric. When p = 1,
#    this is equivalent to using manhattan_distance (l1),
#    and euclidean_distance (l2) for p = 2.
#    For arbitrary p, minkowski_distance (l_p) is used.

params = {
    'n_neighbors': [3, 5, 9, 13, 19, 25, 35, 55, 63],
    'leaf_size': [20, 30, 40, 50, 60],
    'p': [1, 2, 3]
}

knn = KNeighborsClassifier()
knn_clf = GridSearchCV(knn, params, cv=5, scoring=scoring)
knn_clf = knn_clf.fit(X.todense(), y) # KNN takes dense input in scikit-learn


# In[ ]:


print('Best Estimator')
print(knn_clf.best_estimator_)
print('Best Score')
print(knn_clf.best_score_)
print('Best Params')
print(knn_clf.best_params_)


# In[ ]:


y_true, y_pred = y, cross_val_predict(knn_clf.best_estimator_, X, y, n_jobs=5, cv=5)


# In[ ]:


opt_results['KNeighbors'] = {}
opt_results['KNeighbors']['GridSearchCV'] = knn_clf
opt_results['KNeighbors']['classif_report'] = classification_report(y_true, y_pred)


# In[ ]:


print(classification_report(y_true, y_pred))


# In[ ]:


with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)


# ### MLPClassifier

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


params = {
    'hidden_layer_sizes': [(10,5), (20,10), (20), (30,20), (50,30)],
    'activation': ['tanh', 'relu', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.01, 0.001, 0.1],
    'max_iter': [50, 200, 400]
}

mlp = MLPClassifier()
mlp_clf = GridSearchCV(mlp, params, cv=5, scoring=scoring)
mlp_clf = mlp_clf.fit(X, y)


# In[ ]:


print('Best Estimator')
print(clf.best_estimator_)
print('Best Score')
print(clf.best_score_)
print('Best Params')
print(clf.best_params_)


# In[ ]:


y_true, y_pred = y, cross_val_predict(mlp_clf.best_estimator_, X, y, n_jobs=5, cv=5)


# In[ ]:


opt_results['MLP'] = {}
opt_results['MLP']['GridSearchCV'] = mlp_clf
opt_results['MLP']['classif_report'] = classification_report(y_true, y_pred)


# In[ ]:


print(classification_report(y_true, y_pred))


# In[ ]:


with open(opt_results_path, 'wb') as file_:
    pickle.dump(opt_results, file_, protocol=pickle.HIGHEST_PROTOCOL)
