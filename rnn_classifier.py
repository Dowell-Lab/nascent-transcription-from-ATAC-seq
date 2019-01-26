
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.layers import (CuDNNGRU, Embedding, Dense, 
                          Dropout, Bidirectional, concatenate)
from keras.utils import multi_gpu_model


def RNNSeqClassifier(vocab_size, input_length, embedding_dim=32,
                     rnn_hidden_size=100, lr=0.001, dropout=0.1, nr_gpus=1):
    """
    Setup and compile keras RNN nucleotide sequence classifier
    """

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_dim,
                        input_length=input_length,))
    model.add(Dropout(dropout))
    model.add(Bidirectional(CuDNNGRU(units=rnn_hidden_size,
                                     return_sequences=False)))
    model.add(Dense(1, activation="sigmoid")) 

    model.compile(loss='binary_crossentropy', optimizer=Adam(amsgrad=True, lr=lr),
                  metrics=['accuracy'])
    return model


def RNNSigClassifier(input_length, rnn_hidden_size=100, lr=0.001, nr_gpus=1):
    """
    Setup and compile keras RNN signal sequence classifier
    """

    model = Sequential()
    model.add(Bidirectional(CuDNNGRU(rnn_hidden_size,
                                     input_shape=(input_length, 1),
                                     return_sequences=False)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(amsgrad=True, lr=lr),
                  metrics=['accuracy'])
    return model


def RNNHybridClassifier(vocab_size, input_length, embedding_dim=32,
                        rnn_hidden_size=100, lr=0.001, nr_gpus=1):
    """
    Use both sequence and signal as inputs for our RNN
    """

    # one input layer for each source of data
    sig_inp = Input(shape=(input_length, 1))

    seq_inp = Input(shape=(input_length,))
    seq_embed = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim,
                          input_length=input_length)(seq_inp)

    # stack the features on top of each other
    merged = concatenate([sig_inp, seq_embed], axis=2)

    x = Bidirectional(CuDNNGRU(units=rnn_hidden_size,
                               return_sequences=False))(merged)
    
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[sig_inp, seq_inp], outputs=out)
    if nr_gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=nr_gpus)
        parallel_model.compile(loss='binary_crossentropy',
                               optimizer=Adam(amsgrad=True, lr=lr),
                               metrics=['accuracy'])
        return parallel_model
    else:
        model.compile(loss='binary_crossentropy', optimizer=Adam(amsgrad=True, lr=lr),
                      metrics=['accuracy'])
        return model

