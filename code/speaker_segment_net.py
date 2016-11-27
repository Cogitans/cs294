import numpy as np 
import os
# import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.preprocessing import sequence
from data_gen import *


DATA = """FIX"""
TEXT8 = """FIX"""
MODEL_PATH = """FIX"""


BATCH_SIZE = 32
NUM_SAMPLES = 32
TIMESTEPS = 64
BATCH_PER_EPOCH = 1
NUM_EPOCH = 500


def inf_generator(generator):
    while True:
        try:
            for elem in generator():
                yield elem
        except StopIteration:
            continue



def word_mapping(raw_generator):
    """
    Returns a tuple of:
        - distince # of words in dataset
        - mapping from word -> index
        - mapping from index -> word
    """
    total_words = 1
    word_to_idx = {}
    idx_to_word = {}
    for line in raw_generator():
        sentence = line[0].lower()
        for word in sentence.split(' '):
            if word not in word_to_idx:
                    word_to_idx[word] = total_words
                    idx_to_word[total_words] = word
                    total_words += 1
    return total_words - 1, word_to_idx, idx_to_word

def one_hot(text, mapping):
    """
    Takes in a matrix of characters and outputs a matrix of one-hot indexes of same shape
    according to the word -> index dictionary "mapping"
    """
    data = np.zeros(len(text))
    for i in range(len(text)):
        data[i] = mapping[text[i]]
    return data

def shakespeare_soft_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in inf_generator(shakespeare_soft_get):
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))

        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]
        print(Y.shape, targets.shape)
        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def wilde_soft_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in inf_generator(wilde_soft_gen):
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))

        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]
        print(Y.shape, targets.shape)
        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def build_model(input_dim):
    HIDDEN_DIM = 128
    LEARNING_RATE = 3e1
    adam = Adam(lr=LEARNING_RATE)

    model = Sequential()
    model.add(Embedding(input_dim + 1, HIDDEN_DIM, batch_input_shape=(BATCH_SIZE, TIMESTEPS)))
    model.add(GRU(HIDDEN_DIM, return_sequences=True, stateful=True))
    model.add(GRU(2, activation='softmax', return_sequences=True, stateful=True))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model 

num_words, word2idx, indx2word = word_mapping(shakespeare_raw_gen)
train_generator = shakespeare_soft_target_generator(word2idx)
model = build_model(num_words)

history = model.fit_generator(train_generator, BATCH_SIZE, NUM_EPOCH, verbose=2)
# model.save(MODEL_PATH)
# loss = history.history['loss']
# plt.plot(np.arange(len(loss)), loss)
# plt.show()
# plt.savefig("""FIX""")


