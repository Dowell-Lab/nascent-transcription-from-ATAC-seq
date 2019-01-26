import numpy as np
import pywt
import pickle
from sklearn.model_selection import train_test_split
import argparse
from scipy import stats
import supervised_learning
import random


NR_THREADS = 14
#WAVELETS = ['bior6.8', 'cgau8', 'cmor', 'coif17', 'db38', 'dmey', 'fbsp', 'gaus8', 'haar', 'mexh', 'morl', 'rbio6.8', 'shan', 'sym20']
#WAVELETS = pywt.wavelist(kind='discrete')
WAVELETS = ['db1']
OUTPUT_PREFIX = './'

# EIIP represents the distribution of free electrons' energies along the DNA sequence:
EIIP = [0.1260, 0.1340, 0.1335, 0.0806]   # A, C, T, G


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--peak-files', dest='peak_files', help='List of ATAC-seq peak files separated by a pipe (|). If one file is provided, 20% will be used for testing. If multiple files are passed, the last one will be used for testing.', required=True)
parser.add_argument('-g', '--grid', dest='grid', help='Perform a grid search with cross validation for hyperparameter optimization.', required=False, action='store_true')
parser.add_argument('-c', '--cross-validation', dest='cross_validation', help='Run 5-fold cross-validation for the given single file.', required=False, action='store_true')
parser.add_argument('-f', '--classifier', dest='classifier', help='Classifier used. Currently supported: adaboost, rf, svm. Defaults to all of them if not provided.', required=False, default='all')
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
            supervised_learning.classify_RF(args.grid, args.cross_validation, \
                                            X_train_w, X_test_w, y_train_w, y_test_w, coordinates, \
                                            False, 'entropy', 1000, \
                                            single_file, test_file_prefix)

        elif args.classifier == 'svm' or args.classifier == 'all':
            print("Wavelet decomposition peak classification using SVM:")
            supervised_learning.classify_SVM( args.grid, args.cross_validation, \
                                              X_train_w, X_test_w, y_train_w, y_test_w, coordinates, \
                                              'rbf', 10, 'auto', \
                                              single_file, test_file_prefix)

# A: 0
# C: 1
# T: 2
# G: 3
def get_base_index(nucleotide):
    base = 0        # A
    if nucleotide == 'c':
        base = 1
    elif nucleotide == 't':
        base = 2
    elif nucleotide == 'g':
        base = 3
    #TODO: special symbols? (R, N, etc)
    #      for now, flip a coin on the possible bases the symbol represents
    elif nucleotide == 'r':     # purine
        base = np.random.choice([0,3])
    elif nucleotide == 'y':     # pyrimidine
        base = np.random.choice([1,2])
    elif nucleotide == 'k':     # keto
        base = np.random.choice([2,3])
    elif nucleotide == 'm':     # amino
        base = np.random.choice([0,1])
    elif nucleotide == 's':     # strong
        base = np.random.choice([1,3])
    elif nucleotide == 'w':     # weak
        base = np.random.choice([0,2])
    elif nucleotide == 'b':
        base = np.random.choice([1,2,3])
    elif nucleotide == 'd':
        base = np.random.choice([0,2,3])
    elif nucleotide == 'h':
        base = np.random.choice([0,1,2])
    elif nucleotide == 'v':
        base = np.random.choice([0,1,3])
    elif nucleotide == 'n':     # any
        base = np.random.choice([0,1,2,3])
    return base


def get_EIIP_sequence(peak_sequence):
    representation = []
    for nucleotide in peak_sequence.lower():
        nucleotide_index = get_base_index(nucleotide)
        representation.append(EIIP[nucleotide_index])
    return representation


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
                EIIP_sequence_representation = get_EIIP_sequence(peak['sequence'])
                X_train.append([EIIP_sequence_representation])
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
                    EIIP_sequence_representation = get_EIIP_sequence(peak['sequence'])
                    X_train.append([EIIP_sequence_representation])
                    if args.random_labels:
                        y_train.append(random.choice([0, 1]))
                    else:
                        y_train.append(peak['bidir_ovlp'])
        feature_dict = pickle.load(open(peak_files[-1], 'rb'))
        for chromosome in feature_dict:
            for peak in chromosome:
                EIIP_sequence_representation = get_EIIP_sequence(peak['sequence'])
                X_test.append([EIIP_sequence_representation])
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

    wavelet_component_prediction('db1')

