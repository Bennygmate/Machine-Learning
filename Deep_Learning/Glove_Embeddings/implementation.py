#  Neural Networks, Deep Learning
# Bennygmate

import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string

batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    data_list = []
    filename = 'reviews.tar.gz'
    if (os.path.exists('reviews.tar.gz')):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
            with tarfile.open(filename, "r") as tarball:
                dir = os.path.dirname(__file__)
                tarball.extractall(os.path.join(dir, 'data2/'))

    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,'data2/neg/*')))
    for f in file_list:
        with open(f, "r") as openf:
            s = openf.read()
            # Remove punctuation
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            # Make lower case
            preprocess = no_punct.lower()
            # Split into words
            words = preprocess.split()
            row_array = []
            for w in words:
                if w in glove_dict:
                    row_array.append(glove_dict[w])
                else:
                    row_array.append(0)
                if (len(row_array) == 40):
                    break
            # Zero padding if less than 40 words
            row_array = np.pad(row_array, (0, 40), 'constant')
            data_list.append(row_array[:40])
    print (len(data_list))
    print (data_list[24999])
    data = np.array(data_list)
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
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")

    word_index_dict = {}
    word_index_dict['UNK'] = 0
    embeddings = np.ndarray(shape=(500001, batch_size), dtype='float32')
    embeddings_list = []
    i = 1
    for line in data:
        load_array = line.split()
        # Sets the word to the 0th value in array
        word = load_array[0] 
        # Other values are the assigned index
        values = np.asarray(load_array[1:], dtype='float32')
        # Put values in row of array
        embeddings[i] = values
        # E.g. word_index_dict["the"] = 0
        word_index_dict[word] = i
        i = i+1
    data.close()
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
    
    # Local Var
    data_type = 'float32'
    num_classes = 2 # Pos/Neg
    hidden_size = 64 #1500???

    # Input placeholder [batch_size, 40]
    input_data = tf.placeholder(tf.int32, shape=[batch_size, 40])
    # Label placeholder [batch_size, 2] - classes of pos or neg
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    # As given
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    data = tf.Variable(tf.zeros([batch_size, 40, 300]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)

    cell = tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)

    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    
    output, state = tf.nn.dynamic_rnn(cell, data, dtype=data_type)

    w = tf.Variable(tf.truncated_normal([hidden_size, num_classes]))
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    output = tf.transpose(output, [1, 0, 2])
    last_state = tf.gather(output, int(output.get_shape()[0]) - 1)
    prediction = (tf.matmul(last_state, w) + b)
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))

    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    #optimizer = tf.train.GradientDescentOptimizer(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss