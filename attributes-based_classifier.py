import numpy as np
import argparse
import pickle
import supervised_learning
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KEEP_MEAN_THRESHOLD = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--peak-files', dest='peak_files', help='List of ATAC-seq peak files separated by a pipe (|). If one file is provided, 20% will be used for testing. If multiple files are passed, the last one will be used for testing.', required=True)
parser.add_argument('-g', '--grid', dest='grid', help='Perform a grid search with cross validation for hyperparameter optimization.', required=False, action='store_true')
parser.add_argument('-c', '--cross-validation', dest='cross_validation', help='Run 5-fold cross-validation for the given single file.', required=False, action='store_true')
parser.add_argument('-d', '--drop-out', dest='drop_out', help='Drop feature N out of training and test. Must provide the numeric index of the column to be dropped.', required=False)
parser.add_argument('-f', '--classifier', dest='classifier', help='Classifier used. Currently supported: adaboost, rf, svm. Defaults to all of them if not provided.', required=False, default='all')
parser.add_argument('-r', '--random-labels', dest='random_labels', help='Use random labels (for baseline calculations).', required=False, action='store_true')

args = parser.parse_args()

peak_files = args.peak_files.split('|')
single_file = True
print('INPUT FILES:')
print(args.peak_files)
if len(peak_files) > 1:
    single_file = False

X_train = []
y_train = []
coordinates = []

if single_file:
    feature_dict = pickle.load(open(peak_files[0], 'rb'))
    for chromosome in feature_dict:
        for peak in chromosome:
            X_train.append([peak['prom_ovlp'], peak['width'], peak['mean_nr_reads'], peak['max_reads'], \
                      peak['min_reads'], peak['dist_from_last_peak'], peak['gc_ratio']])
            if args.random_labels:
                y_train.append(random.choice([0, 1]))
            else:
                y_train.append(peak['bidir_ovlp'])
            coordinates.append([peak['chrom'], peak['start'], peak['end']])
else:
  for peaks_file in peak_files[:-1]:
        feature_dict = pickle.load(open(peaks_file, 'rb'))
        for chromosome in feature_dict:
            for peak in chromosome:
                X_train.append([peak['prom_ovlp'], peak['width'], peak['mean_nr_reads'], peak['max_reads'], \
                          peak['min_reads'], peak['dist_from_last_peak'], peak['gc_ratio']])
                if args.random_labels:
                    y_train.append(random.choice([0, 1]))
                else:
                    y_train.append(peak['bidir_ovlp'])


X_test = []
y_test = []
test_file_prefix = ''
if single_file:
    full_X_train = np.array(X_train)
    full_y_train = np.array(y_train)
    indices = np.arange(len(full_y_train))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(full_X_train, full_y_train, indices, test_size = 0.2)
    test_file_prefix = peak_files[0].split('/')[-1].split('_')[0] + '-attr'
else:
    feature_dict = pickle.load(open(peak_files[-1], 'rb'))
    for chromosome in feature_dict:
        for peak in chromosome:
            X_test.append([peak['prom_ovlp'], peak['width'], peak['mean_nr_reads'], peak['max_reads'], \
                            peak['min_reads'], peak['dist_from_last_peak'], peak['gc_ratio']])
            if args.random_labels:
                y_test.append(random.choice([0, 1]))
            else:
                y_test.append(peak['bidir_ovlp'])
            coordinates.append([peak['chrom'], peak['start'], peak['end']])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_file_prefix = peak_files[-1].split('/')[-1].split('_')[0] + '-attr'

#X_train, X_test, y_train, y_test = train_test_split(X_new, full_y_train_df, test_size = 0.2)
print("Train/test split:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


if args.drop_out:
    dropped_feature = int(args.drop_out)
    print("Leaving feature %s out" % dropped_feature)
    # Parameters determined by hyperparameter optimization
    print("Peak attribute classification using Random Forests:")
    supervised_learning.classify_RF(args.grid, args.cross_validation, \
                                    np.delete(X_train, dropped_feature, axis=1), np.delete(X_test, dropped_feature, axis=1), y_train, y_test, coordinates, \
                                    True, 'gini', 500, \
                                    single_file, test_file_prefix)

    print("Peak attribute classification using Support Vector Machines:")
    supervised_learning.classify_SVM( args.grid, args.cross_validation, \
                                      np.delete(X_train, dropped_feature, axis=1), np.delete(X_test, dropped_feature, axis=1), y_train, y_test, coordinates, \
                                      'rbf', 100, 0.01, \
                                      single_file, test_file_prefix)
else:
    # Parameters determined by hyperparameter optimization
    if args.classifier == 'adaboost' or args.classifier == 'all':
        print("Peak attribute classification using AdaBoost:")
        supervised_learning.classify_AdaBoost( args.grid, args.cross_validation, \
                                          X_train, X_test, y_train, y_test, coordinates, \
                                          400, 'SAMME.R', \
                                          single_file, test_file_prefix)

    elif args.classifier == 'rf' or args.classifier == 'all':
        print("Peak attribute classification using Random Forests:")
        supervised_learning.classify_RF(args.grid, args.cross_validation, \
                                        X_train, X_test, y_train, y_test, coordinates, \
                                        True, 'gini', 500, \
                                        single_file, test_file_prefix)

    elif args.classifier == 'svm' or args.classifier == 'all':
        print("Peak attribute classification using Support Vector Machines:")
        supervised_learning.classify_SVM( args.grid, args.cross_validation, \
                                          X_train, X_test, y_train, y_test, coordinates, \
                                          'rbf', 100, 0.01, \
                                          single_file, test_file_prefix)

