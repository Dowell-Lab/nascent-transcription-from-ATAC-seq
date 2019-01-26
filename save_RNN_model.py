import sys
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from FeatureExtraction import ItemSelector, Reshape, Seq2Ind
from rnn_classifier import RNNHybridClassifier
from sklearn.externals import joblib


NR_GPUS = 1
LR=0.001
HIDDEN_SIZE = 150
print("Learning rate: %f" % LR)
print("Hidden Layer Size: %d" % HIDDEN_SIZE)
# The dataset used is a pickled Pandas dataframe containing one row per ATAC-seq peak,
# a "sequence" field containing a string of nucleotides for the evaluated window, and a
# "signal_features" field containing an array of real values containing the normalized
# coverage over the evaluation window.
DATASET = './combined_dataset_tfitandfstitch.pkl'
TEST ='some-pandas-df-dump.pk'

###############################################################################
## Load Data
print("Loading data...")
train_set = pd.read_pickle(DATASET)


###############################################################################
## Build Features
sigs = FeatureUnion([
    ('sigs', Pipeline([
        ('feature', ItemSelector('signal_features')),
        ('reshape', Reshape()),
    ])),
])

# sequence
S = set(train_set['sequence'])
T = Tokenizer(char_level=True, oov_token='*')
T.fit_on_texts(S)
seqs = FeatureUnion([
    ('seqs', Pipeline([
        ('feature', ItemSelector('sequence')),
        ('to_index', Seq2Ind(T))
    ]))
])

sigs.fit(train_set)
seqs.fit(train_set)
joblib.dump(sigs, 'final-hybrid-sigs.pkl')
joblib.dump(seqs, 'final-hybrid-seqs.pkl')


###############################################################################
## Build Classifier Pipeline
# which hyperparameters?
# rnn_hidden_size, learning_rate (let embedding_dim=108)

callbacks_list = [ReduceLROnPlateau(patience=3),
                  EarlyStopping(patience=4, restore_best_weights=True)]

pipeline = Pipeline([
    ('clf', KerasClassifier(build_fn=RNNHybridClassifier,
                            vocab_size=len(T.word_index),
                            input_length=1000,
                            embedding_dim=108,
                            rnn_hidden_size=HIDDEN_SIZE,
                            lr=LR,
                            batch_size=128,
                            epochs=40,
                            validation_split=0.1,
                            callbacks=callbacks_list,
                            nr_gpus=NR_GPUS,
                            verbose=1,))
])


y_labels = train_set['bidir_ovlp'].values
###############################################################################
## Fit classifier and evaluate
pipeline.fit(X=[sigs.transform(train_set), seqs.transform(train_set)],
             y=y_labels)


test_set = pd.read_pickle(TEST)
y_pred = pipeline.predict([sigs.transform(test_set), seqs.transform(test_set)])
print(classification_report(test_set['bidir_ovlp'].values, y_pred))
pickle.dump(y_pred, open('test1.pkl', 'wb'))

model = pipeline.named_steps['clf'].model
model.save('final-hybrid-RNN-model.h5')

