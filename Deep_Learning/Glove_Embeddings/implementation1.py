# Neural Networks, Deep Learning
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
            # Get first fourty
            fourty_words = []
            for w in words:
                fourty_words.append(w)
                if (len(fourty_words) == 40):
                    break
            # Fill review array with hash values
            review_array = np.array([])
            for w in fourty_words:
                if w in glove_dict: #KNOWN WORDS
                    review_array = np.append(review_array, glove_dict[w])
                else: #UNKOWN WORDS
                    review_array = np.append(review_array, glove_dict["UNK"])
            # If less than 40 words, fill with zeros
            if (len(review_array) != 40):
                while (len(review_array) != 40):
                    review_array = np.append(review_array, 0)
            data_list.append(review_array)
    print (len(data_list))
    #print (data_list[24999])
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
    embeddings_list = []
    i = 0
    for line in data:
        load_array = line.split()
        # Sets the word to the 0th value in array
        word = load_array[0] 
        # Other values are the assigned index
        values = np.asarray(load_array[1:], dtype='float32')
        # Put values in row of array
        embeddings_list.append(values)
        # E.g. word_index_dict["the"] = 0
        word_index_dict[word] = i
        i = i+1
    data.close()
    # Convert to numpy array
    embeddings = np.array(embeddings_list)
    # 0 Vector for Unknown Words
    word_index_dict["UNK"] = []
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
    data_type = 'float32'
    vocab_size = len(glove_embeddings_arr)
    num_steps = 1
    hidden_size = 1500
    num_layers = 2
    embedding_size = 100

    # EMBEDDINGS FROM PREVIOUS FUNCTION
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    #embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=data_type)
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 40])
    input_data = tf.nn.embedding_lookup(glove_embeddings_arr, train_inputs)
    input_data = tf.nn.dropout(input_data, dropout_keep_prob)
    print (input_data)

    # BASIC 
    #cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
    # BLOCK 
    cell = tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
    #state = initial_state = cell.zero_state(batch_size, data_type)
    state = intitial_state = tf.zeros([batch_size, lstm.state_size])

    output = ""
    outputs = []
    #with tf.variable_scope("RNN"):
    for time_step in range(num_steps):
        print (time_step, "HIII")
        if (time_step > 0):
            tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
            # TRYING
        #inputs = tf.unstack(input_data, num=num_steps, axis = 1)
        output, state = tf.contrib.rnn.static_rnn(cell, input_data, initial_state=initial_state)
        print (output , "WHAT")
    #output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

    # DIFF VERSION
    #input_data = tf.unstack(input_data, num=num_steps, axis=1)
    #outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=initial_state)
    #output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

    softmax_w = tf.get_variable("softmax_w", [embedding_size, vocab_size], dtype=data_type)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type)

    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    # RESHAPE to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
    loss = tf.contrib.seq2seq.sequence_loss(logits, labels, tf.ones([batch_size, num_steps], dtype='float32'), average_across_timesteps=False, average_across_batch=True)


    print (input_data)
    
    lr = tf.variable(0.0, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(lr)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss