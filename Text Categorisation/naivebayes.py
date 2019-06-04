import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import sys
import os
import glob


def main():
    init()
    # get the dataset
    path = "dataset_process"
    main_test(path)


def main_test(path):
    dir_path = path
    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path)
    # calculate the BOW representation
    print colored('Calculating BOW', 'green', attrs=['bold'])
    word_counts = bagOfWords(files.data)
    print(word_counts)

    # TFIDF
    print colored('Calculating TFIDF', 'green', attrs=['bold'])
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    print '\n\n'

    # create classifier
    # clf = sklearn.naive_bayes.MultinomialNB()
    # clf = sklearn.svm.LinearSVC()
    n_neighbors = 11
    weights = 'uniform'
    weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    print '\n\n'
    print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
    test_classifier(X, files.target, clf, test_size=0.2, y_names=files.target_names, confusion=False)


def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print colored('Classification report:', 'magenta', attrs=['bold'])
        print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
    else:
        print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)

def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """
    # Converts a list of strings to a BOW representation
    # @param 'files_data' as a list of strings
    # @return scipy.sparse.coo_matrix
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)

main()