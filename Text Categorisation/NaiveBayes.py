import re
import os
import math
import string
from collections import Counter
from prettytable import PrettyTable

vocab=dict()
vocab_MI=dict()
prob=dict() #probability for p(w|c)
prob_class=dict() # prob for class
total_distinctwordpos=dict()
IG=dict()
#Train_path = "20news-bydate-train"
Train_path = "cross_valid/10/train"
#Test_path = "20news-bydate-test"
Test_path = "cross_valid/10/test"
# FEATURE SELECTION VARIABLES
least_wordfrequency = 2
mostcommon_words = 100
simple_average_common = 5000


def main():
    # Training folder
    Train = Train_path
    # V is the set of all possible target values.
    V = os.listdir(Train)
    learn_naive_bayes_text(Train, V)
    naive_bayes_test(V)

def naive_bayes_test(V):
    print "Split of 60% training and 40% testing"
    total_accuracy = 0
    # Number of items correctly labelled as belonging to positive class
    true_pos=dict()
    # Items incorrectly labelled as belonging to the class
    false_pos=dict()
    # Items which were not labelled as belonging to positive class but should have been
    false_neg=dict()
    # Precision - true positive / (true pos + false pos)
    precision=dict()
    # Recall - true pos / (true pos + false neg)
    recall=dict()
    # F1 = 2 * ((precision * recall) / (precision + recall))
    f1_score=dict()
    for vj in V:
        true_pos[vj] = 0
        false_pos[vj] = 0
        false_neg[vj] = 0
        precision[vj] = 0
        recall[vj] = 0
        f1_score[vj] = 0
    for vj in V:
        correct_classification = 0
        path, direc, docs = os.walk(Test_path+"/"+vj).next()
        for doc_name in docs:
            doc_path = "" + path + "/" + doc_name
            target_value = classify_naive_bayes_text(doc_path, V)
            if target_value == vj:
                true_pos[vj] += 1
            else:
                false_neg[vj] += 1
                false_pos[target_value] += 1

    for vj in V:
        precision[vj] = float(true_pos[vj]) / (true_pos[vj] + false_pos[vj])
        recall[vj] = float(true_pos[vj]) / (true_pos[vj] + false_neg[vj])
        f1_score[vj] = 2 * ((precision[vj] * recall[vj]) / (precision[vj] + recall[vj]))

    space = ""
    print "Naive Bayes %f" %(sum(recall.values()) / 20)
    print "%26sprecision%4srecall%2sf1-score%3ssupport\n" %(space, space, space, space)
    for vj in V:
        print "%24s%7s%.2f%6s%.2f%6s%.2f%7s%d" %(vj, space, round(precision[vj], 2),
                                                space, round(recall[vj], 2), 
                                                space, round(f1_score[vj],2),
                                                space, true_pos[vj])
    avg = "avg / total"
    print "\n%24s%7s%.2f%6s%.2f%6s%.2f%7s%d" %(avg, space, round((sum(precision.values()) / 20), 2), 
                                             space, round((sum(recall.values()) / 20), 2), 
                                             space, round((sum(f1_score.values()) / 20), 2),
                                             space, (sum(true_pos.values()) / 20))


def classify_naive_bayes_text(Doc, V):
    #Calculate probabilities for words in a document belonging to a class
    # Initialise test_prob to multiply later for arg_max of class
    test_prob = dict()
    # Grab classes
    for vj in V:
        test_prob[vj] = 0
    with open(Doc, 'r') as docsj:
        for textj in docsj:
            words = tokenize(textj)
            for word in words:
                if word in vocab:
                    # Then this meets the position
                    # positions - all word positions in Doc that contain tokens found in vocabulary
                    for vj in V:
                        # Apparently there is a problem with floats tending to 0 when reaching e-320
                        # Looked at internet for help and apparently sum the logarithms instead
                        # when comparing which would be larger, some math theorem i don't understand
                        # see https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
                        if (test_prob[vj] == 0):
                            #test_prob[vj] = prob[vj][word] * prob_class[vj]
                            test_prob[vj] = math.log(prob[vj][word]+IG[vj][word]) + math.log(prob_class[vj])
                        else:
                            #test_prob[vj] = test_prob[vj] * prob[vj][word]
                            test_prob[vj] += math.log(prob[vj][word]+IG[vj][word])

    # Take the argmax P(vj)
    max_prob = -1000000000
    for vj in test_prob:
        if test_prob[vj] > max_prob:
            max_prob = test_prob[vj]
            vNB = vj
    # Return the estimated target value for the document Doc
    return vNB

def learn_naive_bayes_text(Examples, V):
    # Examples is a set of text documents along with their target values (training data)
    print "Building initial vocabulary and calculating probabilities"
    print "========================================================="
    # For each vj in V do
    for vj in V:
        # Grab path to file and docs
        path, direc, docs = os.walk(Examples+"/"+vj).next() # seperating files from folders
        # docsj
        prob_class[vj] = len(docs)
        prob[vj], total_distinctwordpos[vj] = build_vocabulary(path, docs)

    # Vocabulary is created
    print "Vocabulary is created with length %d" %len(vocab)
    print "========================================================="

    # P(vj) = |docsj| / |Examples| (The class prior probabilties)
    total_examples = sum(prob_class.values())
    for vj in V:
        prob_class[vj] = float(prob_class[vj]) / total_examples

    feature_selection(V)

    #Calculating probabilities for each class and its words.
    print "====================Training  Classes===================="
    print V
    for vj in V:
        n_words = total_distinctwordpos[vj]
        words_vocab = len(vocab)
        total_count = n_words + words_vocab
        for word in vocab:
            if word in prob[vj]:
                count = prob[vj][word]
            else:
                count = 0
            # P(wk|vj) = (nk + 1) / (n + |Vocabulary|)
            prob[vj][word] = float(count+1)/total_count
    print "====================Training Finished===================="

def build_vocabulary(path, docs):
    # Counting words and building vocab
    # nk_wordk - number of times word wk occurs in Textj
    nk_wordk=dict()
    # n_words - total number of distinct word positions in Textj (add for all text)
    n_words = 0
    # docsj - the subset of documents from Examples for which the target value is vj
    for doc in docs:
        with open(path+'/'+doc,'r') as docsj:
            for textj in docsj:
                #field = re.match(r'From:', textj)
                #if (field):
                #    continue
                #field = re.match(r'Organization:', textj)
                #if (field):
                #    continue
                #field = re.match(r'Lines:', textj)
                #if (field):
                #    continue
                words = tokenize(textj)
                for word in words:
                    # Remove digit words
                    field = re.match(r'.*(\d+).*', word)
                    if (field):
                        continue
                    if word not in vocab:
                        vocab[word] = 1 # Add to vocab
                        nk_wordk[word] = 1 # Add to count
                    elif word != '':
                        nk_wordk.setdefault(word, 0)
                        vocab[word] += 1
                        nk_wordk[word] += 1
                    n_words += 1
    return nk_wordk, n_words

def feature_selection(V):
    print "Feature Selection:"
    print "1. Pruning words with frequency less than %d" %least_wordfrequency
    # Deleting less frequent words with count less than 3
    infrequent_words()
    print "2. Pruning most %d common words" %mostcommon_words
    # Deleting 100 most frequent examples
    high_frequency_words()
    print "3. Choosing words with high mutual information with class"
    mutual_information(V)
    print "========================================================="

def mutual_information(V):
    prob_MI=dict()
    for vj in V:
        if vj not in prob_MI:
            prob_MI[vj] = {}
        n_words = total_distinctwordpos[vj]
        words_vocab = len(vocab)
        total_count = n_words + words_vocab
        for word in vocab:
            if word in prob[vj]:
                count = prob[vj][word]
            else:
                count = 0
            # P(wk|vj) = (nk + 1) / (n + |Vocabulary|)
            prob_MI[vj][word] = float(count)/total_count

    for vj in V:
        if vj not in IG:
            IG[vj] = {}
        for word in vocab:
            class_log = math.log(prob_class[vj])
            Entropy_C = (prob_class[vj] * class_log)
            notword_log = math.log((1 - prob_MI[vj][word]) * prob_class[vj])
            Entropy_NW = ((1-prob_MI[vj][word]) * prob_class[vj]) * notword_log
            if prob_MI[vj][word] == 0:
                word_log = 0
            else:
                word_log = math.log(prob_MI[vj][word] * prob_class[vj])
            Entropy_W = prob_MI[vj][word] * prob_class[vj] * word_log
            if word not in IG[vj]:
                IG[vj][word] = Entropy_C - Entropy_NW - Entropy_W
                #print IG[vj][word]
            else:
                IG[vj][word] += Entropy_C - Entropy_NW - Entropy_W
    # OPTIONAL TO AVERAGE COMMON INFORMATION GAIN, to help discriminate
    if simple_average_common != 0:
        vocab_mostcommon=dict()
        common_word=dict()
        for vj in V:
            d = Counter(IG[vj]).most_common(simple_average_common)
            i = 0
            while (i < simple_average_common):
                word = d[i][0]
                if word not in vocab_mostcommon:
                    vocab_mostcommon[word] = 0
                else:
                    if word not in common_word:
                        common_word[word] = 0
                i += 1
        for word in common_word.keys():
                average = 0
                counter = 0
                for vj in V:
                    if IG[vj][word] > 0:
                        average += IG[vj][word]
                        counter += 1
                for vj in V:
                    if IG[vj][word] > 0:
                        IG[vj][word] = average / counter
        print "Simple Average for %d common words mutual in class" %(simple_average_common)
    print "Vocabulary for information gain complete"


def infrequent_words():
    mark_del = []
    for word in vocab:
        if vocab[word] < least_wordfrequency:
                mark_del.append(word)
    for word in mark_del:
        del vocab[word]
    print "Vocabulary is pruned with length %d" %len(vocab)

def high_frequency_words():  
    mark_del = []
    d = Counter(vocab).most_common(mostcommon_words)
    for word in vocab:
        i = 0
        while (i < mostcommon_words):
            if word == d[i][0]:
                mark_del.append(word)
            i += 1
    for word in mark_del:
        del vocab[word]        
    print "Vocabulary is pruned with length %d" %len(vocab)

def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    return re.split("\W+", text)

def remove_punctuation(s):
    table = string.maketrans("","")
    return s.translate(table, string.punctuation)

main()

