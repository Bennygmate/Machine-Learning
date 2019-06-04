import sys
import sklearn.datasets
import sklearn.metrics
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# TUTORIAL http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Defining Categories which are included in the dataset from 20newsgroups dataset
categories = os.listdir("20news-bydate-train")
# Load the data matching the categories 
# TRAIN PATH "cross_valid/10/train"
twenty_train = sklearn.datasets.load_files(container_path= r'20news-bydate-train', encoding='latin1')
#twenty_train = sklearn.datasets.load_files(container_path= r'cross_valid/10/train', encoding='latin1')
# Number of documents
print "Training on 20 classes:", twenty_train.target_names

# Text preprocessing, tokenizing, filtering of stopwords
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print "(Number of documents, Vocabulary Length):", (X_train_counts.shape)

# Computing Term Frequencies from bag of words, inverse document frequency option off
tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)

length = len(twenty_train.data)
# TEST PATH
twenty_test = sklearn.datasets.load_files(container_path= r'20news-bydate-test', encoding= 'latin1')
#twenty_test = sklearn.datasets.load_files(container_path= r'cross_valid/10/test', encoding= 'latin1' )

docs_test = twenty_test.data

value = []
i = -1
for classifier_type in range(1,5):
    i = i+1
    value.append([])
    if(classifier_type == 1):
        print("Using Naive Bayes")
        # Making vector --> transformed --> classifier through a pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer(use_idf=True)), #IDF ON OR OFF
                            ('clf', MultinomialNB()),
                            ])
    elif(classifier_type == 2):
        print("Using Logistic Regression")
        # Making vector --> transformed --> classifier through a pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])
    elif(classifier_type == 3):
        print("Using Linear SVC Classifier")
        # Making vector --> transformed --> classifier through a pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LinearSVC())])
    elif(classifier_type == 4):
        print("Using Random Forest Classifier")
        # Making vector --> transformed --> classifier through a pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', RandomForestClassifier())])
        
    # Training the model with a single command
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
    # Evaluating the model
    predicted = text_clf.predict(docs_test)
    # Printing the statistics
    print(np.mean(predicted == twenty_test.target))
    # Utility for detailed performance analysis
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

