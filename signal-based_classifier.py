import numpy as np
import pywt
import multiprocessing
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.optimizers import RMSprop
import argparse
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import supervised_learning


NR_THREADS = 14
#WAVELETS = ['bior6.8', 'cgau8', 'cmor', 'coif17', 'db38', 'dmey', 'fbsp', 'gaus8', 'haar', 'mexh', 'morl', 'rbio6.8', 'shan', 'sym20']
#WAVELETS = pywt.wavelist(kind='discrete')
WAVELETS = ['db38']
OUTPUT_PREFIX = './'



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--peak-files', dest='peak_files', help='List of ATAC-seq peak files separated by a pipe (|). If one file is provided, 20% will be used for testing. If multiple files are passed, the last one will be used for testing.', required=True)
parser.add_argument('-g', '--grid', dest='grid', help='Perform a grid search with cross validation for hyperparameter optimization.', required=False, action='store_true')
parser.add_argument('-c', '--cross-validation', dest='cross_validation', help='Run 5-fold cross-validation for the given single file.', required=False, action='store_true')
parser.add_argument('-f', '--classifier', dest='classifier', help='Classifier used. Currently supported: adaboost, rf, svm. Defaults to all of them if not provided.', required=False, default='all')
parser.add_argument('-e', '--encoding', dest='encoding', help='Encoding used for the signal. Currently supported: wavelet, autoencoder', required=True)
parser.add_argument('-r', '--random-labels', dest='random_labels', help='Use random labels (for baseline calculations).', required=False, action='store_true')
args = parser.parse_args()


def wavelet_component_prediction(current_wavelet):
    mode = 'periodic'
    #for mode in ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect']:
    #for level in [1, 2, 3]:
    for level in [1]:
        level = 1
        print("Using the %s wavelet with mode %s, level %d" % (current_wavelet, mode, level))
        X_train_w = []
        X_test_w = []
        y_train_w = []
        y_test_w = []

        if single_file:
            full_X_train = []
            for i in range(X_train.shape[0]):
                full_X_train.append(pywt.wavedec(X_train[i], current_wavelet, mode=mode, level=level)[0])

            full_X_train = np.array(full_X_train)
            full_y_train = np.array(y_train)
            indices = np.arange(len(y_train))
            X_train_w, X_test_w, y_train_w, y_test_w, idx_train, idx_test = \
                train_test_split( full_X_train, \
                                  full_y_train, \
                                  indices, \
                                  test_size = 0.2)
        else:
            for i in range(len(y_train)):
                X_train_w.append(pywt.wavedec(X_train[i], current_wavelet, mode=mode, level=level)[0])
            X_train_w = np.array(X_train_w)
            for i in range(X_test.shape[0]):
                X_test_w.append(pywt.wavedec(X_test[i], current_wavelet, mode=mode, level=level)[0])
            X_test_w = np.array(X_test_w)
            y_train_w = y_train
            y_test_w = y_test

        if args.classifier == 'adaboost' or args.classifier == 'all':
            print("Wavelet decomposition peak classification using AdaBoost:")
            supervised_learning.classify_AdaBoost( args.grid, args.cross_validation, \
                                                  X_train_w, X_test_w, y_train_w, y_test_w, coordinates, \
                                                  400, 'SAMME.R', \
                                                  single_file, test_file_prefix)

        elif args.classifier == 'rf' or args.classifier == 'all':
            print("Wavelet decomposition peak classification using Random Forests:")
            # Parameters determined by hyperparameter optimization
            supervised_learning.classify_RF(args.grid, args.cross_validation, \
                                            X_train_w, X_test_w, y_train_w, y_test_w, coordinates, \
                                            False, 'gini', 1000, \
                                            single_file, test_file_prefix)

        elif args.classifier == 'svm' or args.classifier == 'all':
            print("Wavelet decomposition peak classification using SVM:")
            supervised_learning.classify_SVM( args.grid, args.cross_validation, \
                                              X_train_w, X_test_w, y_train_w, y_test_w, coordinates, \
                                              'rbf', 10, 'auto', \
                                              single_file, test_file_prefix)



def autoencoded_predictions(encoding_dim):
    print("Using Autoencoder to reduce the signal-based features to %d dimensions" % encoding_dim)
    input_signal = Input(shape=(1000,))
    encoded = Dense(encoding_dim, activation='relu')(input_signal)
    decoded = Dense(1000, activation='sigmoid')(encoded)
    encoder = Model(input_signal, encoded)
    autoencoder = Model(input_signal, decoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.summary()
    autoencoder.fit(X_train, X_train, epochs=50, \
                    batch_size=256, shuffle=True)
    X_train_e = encoder.predict(X_train)
    X_test_e = encoder.predict(X_test)
    print("Sample of new features:")
    print(X_train_e[0])

    if args.classifier == 'adaboost' or args.classifier == 'all':
        print("\n\nAutoencoded peak classification using AdaBoost:")
        supervised_learning.classify_AdaBoost( args.grid, args.cross_validation, \
                                              X_train_e, X_test_e, y_train, y_test, coordinates, \
                                              400, 'SAMME.R', \
                                              single_file, test_file_prefix)

    elif args.classifier == 'rf' or args.classifier == 'all':
        print("Autoencoded peak classification using Random Forests:")
        supervised_learning.classify_RF(args.grid, args.cross_validation, \
                                        X_train_e, X_test_e, y_train, y_test, coordinates, \
                                        False, 'gini', 1000, \
                                        single_file, test_file_prefix)

    elif args.classifier == 'svm' or args.classifier == 'all':
        print("\n\nAutoencoded peak classification using Support Vector Machines:")
        supervised_learning.classify_SVM( args.grid, args.cross_validation, \
                                          X_train_e, X_test_e, y_train, y_test, coordinates, \
                                          'rbf', 10, 'auto', \
                                          single_file, test_file_prefix)


if __name__=='__main__':
    peak_files = args.peak_files.split('|')
    single_file = True
    print('INPUT FILES:')
    print(args.peak_files)
    if len(peak_files) > 1:
          single_file = False

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    coordinates = []
    test_file_prefix = ''
    if single_file:
        feature_dict = pickle.load(open(args.peak_files[0], 'rb'))
        for chromosome in feature_dict:
            for peak in chromosome:
                X_train.append([peak['signal_features']])
                if args.random_labels:
                    y_train.append(random.choice([0, 1]))
                else:
                    y_train.append(peak['bidir_ovlp'])
                coordinates.append([peak['chrom'], peak['start'], peak['end']])
        test_file_prefix = peak_files[0].split('/')[-1].split('_')[0] + '-sig'
    else:
        # The last file listed is used for testing
        for peaks_file in peak_files[:-1]:
            feature_dict = pickle.load(open(peaks_file, 'rb'))
            for chromosome in feature_dict:
                for peak in chromosome:
                    X_train.append([peak['signal_features']])
                    if args.random_labels:
                        y_train.append(random.choice([0, 1]))
                    else:
                        y_train.append(peak['bidir_ovlp'])
        feature_dict = pickle.load(open(peak_files[-1], 'rb'))
        for chromosome in feature_dict:
            for peak in chromosome:
                X_test.append([peak['signal_features']])
                if args.random_labels:
                    y_test.append(random.choice([0, 1]))
                else:
                    y_test.append(peak['bidir_ovlp'])
                coordinates.append([peak['chrom'], peak['start'], peak['end']])
        test_file_prefix = peak_files[-1].split('/')[-1].split('_')[0] + '-sig'

    X_train = np.array(X_train)
    X_train = X_train.reshape(len(y_train), 1000)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], 1000)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if args.encoding == 'wavelet':
        wavelet_component_prediction('db38')
    elif args.encoding == 'autoencoder':
        autoencoded_predictions(15)

