import pickle
import sys
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd


SIGS_PICKLE = './final-hybrid-sigs.pkl'
SEQS_PICKLE = './final-hybrid-seqs.pkl'
MODEL = './final-hybrid-RNN-model.h5'
ATAC_PEAKS_PICKLE = sys.argv[1]
OUTPUT_PEAK_BEDFILE = sys.argv[2]

model=load_model(MODEL)
test_set = pd.read_pickle(ATAC_PEAKS_PICKLE)

sigs = joblib.load(SIGS_PICKLE)
seqs = joblib.load(SEQS_PICKLE)

y_pred = model.predict([sigs.transform(test_set), seqs.transform(test_set)])
y_pred = np.where(y_pred > 0.5, 1, 0)

print(classification_report(test_set['bidir_ovlp'].values, y_pred))

pickle.dump(y_pred, open(sys.argv[1] + '-pred.pkl', 'wb'))

test_set['prediction'] = y_pred
positives = test_set[test_set['prediction'] > 0.5]
with open(OUTPUT_PEAK_BEDFILE ,'w') as pred_file:
    for row in positives.itertuples():
        pred_file.write("%s\t%s\t%s\n" % (row.chrom, row.start, row.end))

