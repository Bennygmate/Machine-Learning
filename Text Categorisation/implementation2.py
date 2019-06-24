import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    arr = []

    tar = tarfile.open("reviews.tar.gz", "r:gz")
    for i, member in enumerate(tar.getmembers()):
        f = tar.extractfile(member)
        data = f.readline()

        #PREPROCESSING
        # convert to lowercase:
        data = data.lower() # convert to lowercase
        
        # remove punctuation:
        allowed_characters = set("abcdefghijklmnopqrstuvwxyz0123456789 ") # todo: space might be issue
        data = ''.join(ch for ch in data if ch in allowed_characters)

        # strip out unnecessary words:
        # unnecessary_words = ["a", "about", "at", "the", "all", "are", "as", "at", "but", "by", "did", "do"
        # "during", "each", "had", "has", "he", "she", "they", "it", "i", "is", "than", "that", "their", "them"]
        unnecessary_words = []
        data = ' '.join(word for word in data.split(" ") if word not in unnecessary_words)

        # "vectorize" it:
        row = []
        for word in data.split(" "):
            row.append(glove_dict[word] if word in glove_dict.keys() else 0)

        # zero pad:
        row = np.pad(row, (0, 40), 'constant')

        # add only first 40 words
        arr.append(row[:40])
    data = np.array(arr)

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    word_index_dict = {}
    word_index_dict['UNK'] = 0
    embeddings = np.ndarray(shape=(500001, batch_size), dtype=np.float32)

    with open("glove.6B.50d.txt",'r',encoding="utf-8") as f:
        i = 1
        for line in f.readlines():
            tokens = line.split(" ")
            word = tokens[0]
            word_index_dict[word] = i

            values = [float(v) for v in tokens[1:]]
            data_np = np.asarray(values, np.float32)

            embeddings[i] = data_np
            i += 1

    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    tf.reset_default_graph()

    dropout_keep_prob = tf.placeholder_with_default(0.75, shape=())


    numClasses = 2
    lstmUnits = 64
    maxSeqLength = 40

    labels = tf.placeholder(tf.float32, [batch_size, numClasses], name="labels")
    input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength], name="input_data")

    data = tf.Variable(tf.zeros([batch_size, maxSeqLength, 300]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)

    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    


    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss