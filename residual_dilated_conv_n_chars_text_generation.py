import random
import sys
import numpy as np
from keras.utils import get_file
from keras.models import Model
from keras.layers import Flatten, Dense, Input, merge, Activation, Conv1D, BatchNormalization, RepeatVector, \
    TimeDistributed, Reshape


def prepare_data(file_name, origin, maxlen, step, n_chars_prediction=1):
    path = get_file(file_name, origin)
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - (maxlen + n_chars_prediction), step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen: i + maxlen + n_chars_prediction])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), n_chars_prediction, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        for t, next_char in enumerate(next_chars[i]):
            y[i, t, char_indices[next_char]] = 1
    return X, y, text, char_indices, indices_char, chars


def res_dilated_block_1(filters, dilation_rate, filter_size):
    def layer(input_):
        # if dilation_rate > 1:
        #     input_ = BatchNormalization()(input_)
        input_ = Activation('relu')(input_)
        conv_1 = Conv1D(filters, filter_size, padding='same',
                        activation='relu')(input_)
        conv_1 = Conv1D(filters, filter_size, dilation_rate=dilation_rate, padding='causal',
                        activation='relu')(conv_1)

        if dilation_rate > 1:
            out = merge([conv_1, input_], mode='sum')
        else:
            out = conv_1
        return out, conv_1
    return layer


def res_dilated_conv_model(input_shape, n_chars_prediction):
    max_features = input_shape[1]
    input = Input(input_shape)
    seq = input
    res_layers = []
    for i in range(7):
        seq, curr_layer = res_dilated_block_1(256, 2**i, 5 if i == 0 else 2)(seq)
        res_layers.append(curr_layer)

    seq = Flatten()(seq)
    # seq = RepeatVector(n_chars_prediction)(seq)
    seq = Dense(max_features*n_chars_prediction)(seq)
    seq = Reshape((n_chars_prediction, max_features))(seq)
    seq = TimeDistributed(Activation('softmax'))(seq)
    model = Model(input=input, output=seq)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # taken from keras examples - lstm_text_generation
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    n_epochs = 100
    maxlen = 128
    step = 3
    n_chars_prediction = 2

    X, y, text, char_indices, indices_char, chars = \
        prepare_data('nietzsche.txt', "https://s3.amazonaws.com/text-datasets/nietzsche.txt",
                     maxlen, step, n_chars_prediction)

    model = res_dilated_conv_model((maxlen, len(chars)),n_chars_prediction)

    for i in range(n_epochs):
        model.fit(X, y, batch_size=128, nb_epoch=1)

        start_index = random.randint(0, len(text) - maxlen - 1)

        print()
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(200):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = [sample(preds[i], 0.2) for i in range(n_chars_prediction)]

            next_chars = ''.join([indices_char[next_index[i]] for i in range(n_chars_prediction)])

            generated += next_chars
            sentence = sentence[n_chars_prediction:] + next_chars

            sys.stdout.write(next_chars)
            sys.stdout.flush()
        print()
