import sys
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from FeatureExtraction import ItemSelector, Reshape, Seq2Ind
from rnn_classifier import RNNHybridClassifier


NR_GPUS = 1
LR = float(sys.argv[1])
HIDDEN_SIZE = int(sys.argv[2])
TEST_SAMPLE = sys.argv[3]   # e.g. SRR5876158
LABEL_SOURCE = sys.argv[4]  # e.g. tfitandfstitch, histmarksonly, etc
RANDOM_LABELS = False
if len(sys.argv) > 5:
    RANDOM_LABELS = True
print("Learning rate: %f" % LR)
print("Hidden Layer Size: %d" % HIDDEN_SIZE)
DATA_DIR = './'

# SRR* Samples
datasets = None
test_index = None
datasets = [
    'SRR1822165',
    'SRR1822167',
    'SRR1822166',
    'SRR5876159',
    'SRR5007259',
    'SRR5876158',
    'SRR1822168',
    'SRR5007258',
    'SRR5128074',
]
test_index = datasets.index(TEST_SAMPLE)

###############################################################################
## Load Data
print("Loading data...")
data = None
data = pd.read_pickle('%s/combined_dataset_%s.pkl' % (DATA_DIR, LABEL_SOURCE))


# get a subest for train/test
# Comment out one of these.
# The commented out line should be used as the test set
train_samples = []
for i in range(len(datasets)):
    if i != test_index:
        train_samples.append(datasets[i])

train_set = data.loc[train_samples]

test_dataset = datasets[test_index]
test_set = data.loc[test_dataset]
print("TESTING ON {}".format(TEST_SAMPLE))
print("TRAINING ON:")
print(train_samples)

del data



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
if RANDOM_LABELS:
    y_labels = np.random.randint(low=2, size=len(train_set))
###############################################################################
## Fit classifier and evaluate
pipeline.fit(X=[sigs.transform(train_set), seqs.transform(train_set)],
             y=y_labels)

## Evaluate performance on the test set using the best model
y_pred = pipeline.predict([sigs.transform(test_set), seqs.transform(test_set)])

my_y_test = test_set['bidir_ovlp'].values
if RANDOM_LABELS:
    my_y_test = np.random.randint(low=2, size=len(test_set))
coordinates = test_set[['chrom','start','end']].values
target_names = ['non-functional binding', 'functional binding']

print("-------------------------------------------------------------------")
print("RNN (Hybrid model) Confusion Matrix:")
print(confusion_matrix(my_y_test, y_pred))
print("-------------------------------------------------------------------")
print("F1-score (macro): %.3f" % f1_score(my_y_test, y_pred, average='macro'))
print("F1-score (micro): %.3f" % f1_score(my_y_test, y_pred, average='micro'))
print("F1-score (weighted): %.3f\n" % f1_score(my_y_test, y_pred, average='weighted'))
print("classification_report:")
print(classification_report(my_y_test, y_pred, target_names=target_names))
print("Accuracy: {}".format(accuracy_score(my_y_test, y_pred)))

## Plot ROC and report AUC
probas = pipeline.predict_proba([sigs.transform(test_set), seqs.transform(test_set)])
fpr, tpr, thresholds = roc_curve(my_y_test, probas[:, 1])
print("AUC: {}".format(auc(fpr, tpr)))

#pipeline.named_steps['clf'].model.save('{}-.h5'.format(TEST_SAMPLE))

model = pipeline.named_steps['clf'].model
model_json = model.to_json()
with open("%s/%s-hybrid-RNN-model.json" % (DATA_DIR, TEST_SAMPLE), "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s/%s-hybrid-RNN-model-weights.h5" % (DATA_DIR, TEST_SAMPLE))
    print("Saved %s model to disk" % TEST_SAMPLE)

false_positives = []
false_negatives = []
true_positives = []
true_negatives = []
for i in range(len(y_pred)):
    if my_y_test[i] == 0 and y_pred[i] == 1:
        false_positives.append(coordinates[i])
    elif my_y_test[i] == 1 and y_pred[i] == 0:
        false_negatives.append(coordinates[i])
    elif my_y_test[i] == 0:
        true_negatives.append(coordinates[i])
    else:
        true_positives.append(coordinates[i])

with open("%s/%s-vs-%s_RNN-hybrid_false_positives.bed" % (DATA_DIR, TEST_SAMPLE, LABEL_SOURCE), 'w') as fp_file:
    for peak in false_positives:
        fp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
with open("%s/%s-vs-%s_RNN-hybrid_false_negatives.bed" % (DATA_DIR, TEST_SAMPLE, LABEL_SOURCE), 'w') as fn_file:
    for peak in false_negatives:
        fn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
with open("%s/%s-vs-%s_RNN-hybrid_true_positives.bed" % (DATA_DIR, TEST_SAMPLE, LABEL_SOURCE), 'w') as tp_file:
    for peak in true_positives:
        tp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
with open("%s/%s-vs-%s_RNN-hybrid_true_negatives.bed" % (DATA_DIR, TEST_SAMPLE, LABEL_SOURCE), 'w') as tn_file:
    for peak in true_negatives:
        tn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))



