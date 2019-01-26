import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

# to prevent display weirdness when running in Pando:
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


OUTPUT_PREFIX = './'


def classify_RF(grid, cross_validation, my_X_train, my_X_test, my_y_train, my_y_test, coordinates, bootstrap, criterion, n_estimators, single_file, test_file_prefix):
    print('RF classifier called with args: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (grid, cross_validation, my_X_train.shape, my_X_test.shape, my_y_train.shape, my_y_test.shape, len(coordinates), bootstrap, criterion, n_estimators, single_file, test_file_prefix))
    target_names = ['non-functional binding', 'functional binding']
    # Try Random Forests
    if grid:
        parameters = {  'bootstrap':(True, False), \
                        'criterion':['gini', 'entropy'], \
                        'n_estimators':[500, 1000] }
        print("\nRandom Forests hyperparameter grid search for first wavelet components:")
        rf = RandomForestClassifier()
        clf = GridSearchCV(rf, cv=5, n_jobs=8, param_grid=parameters)
        clf.fit(my_X_train, my_y_train.ravel())
        print("-------------------------------------------------------------------")
        print("RF Best params:")
        print(clf.best_params_)
        print("-------------------------------------------------------------------")
        print("RF Best score:")
        print(clf.best_score_)
        print("-------------------------------------------------------------------")
        print("RF All results:")
        print(clf.cv_results_)
        print("-------------------------------------------------------------------")
    elif cross_validation:
        clf = RandomForestClassifier(bootstrap=bootstrap, criterion=criterion, n_estimators=n_estimators)   # Determined by hyperparameter optimization
        # 5-fold cross-validation
        scores = cross_val_score(clf, my_X_train, my_y_train.ravel(), cv=5, scoring="f1_weighted")
        print(scores)
        print("-------------------------------------------------------------------  ")
        print("Mean weighted F1-score: %f" % np.mean(scores))
        print("Std weighted F1-score: %f" % np.std(scores))
        print("-------------------------------------------------------------------  ")
    else:
        clf = RandomForestClassifier(bootstrap=bootstrap, criterion=criterion, n_estimators=n_estimators)   # Determined by hyperparameter optimization
        print(clf)
        clf.fit(my_X_train, my_y_train.ravel())
        y_pred = clf.predict(my_X_test)

        # Store the classifier to reuse later
        pickle.dump(clf, open("%s/%s_RF_clf.pickle" % (OUTPUT_PREFIX, test_file_prefix), 'wb'))

        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        if single_file:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[idx_test[i]])
                else:
                    true_positives.append(coordinates[idx_test[i]])
        else:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[i])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[i])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[i])
                else:
                    true_positives.append(coordinates[i])

        with open("%s/%s_RF_false_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fp_file:
            for peak in false_positives:
                fp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_RF_false_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fn_file:
            for peak in false_negatives:
                fn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_RF_true_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tp_file:
            for peak in true_positives:
                tp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_RF_true_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tn_file:
            for peak in true_negatives:
                tn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))

        print("-------------------------------------------------------------------")
        print("RF Confusion Matrix:")
        print(confusion_matrix(my_y_test, y_pred))
        print("-------------------------------------------------------------------")
        print("F1-score (macro): %.3f" % f1_score(my_y_test, y_pred, average='macro'))
        print("F1-score (micro): %.3f" % f1_score(my_y_test, y_pred, average='micro'))
        print("F1-score (weighted): %.3f\n" % f1_score(my_y_test, y_pred, average='weighted'))
        print(classification_report(my_y_test, y_pred, target_names=target_names))
        print("-------------------------------------------------------------------")
        print("Accuracy: {}".format(accuracy_score(my_y_test, y_pred)))

#        # Plot ROC and report AUC
#        probas = clf.predict_proba(my_X_test)
#        fpr, tpr, thresholds = roc_curve(my_X_test, probas[:, 1])
#        print("AUC: {}".format(auc(fpr, tpr)))
#
#        plt.plot(fpr, tpr);
#        plt.title('RF Receiver Operating Characteristic')
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.axis('equal')
#        plt.xlim((0, 1));
#        plt.ylim((0, 1));
#        plt.plot(range(2), range(2), '--');
#        plt.savefig("%s/%s_RF_ROC.png" % (OUTPUT_PREFIX, test_file_prefix), dpi=600)

def classify_SVM(grid, cross_validation, my_X_train, my_X_test, my_y_train, my_y_test, coordinates, kernel, C, gamma, single_file, test_file_prefix):
    print('SVM classifier called with args: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (grid, cross_validation, my_X_train.shape, my_X_test.shape, my_y_train.shape, my_y_test.shape, len(coordinates), kernel, C, gamma, single_file, test_file_prefix))
    target_names = ['non-functional binding', 'functional binding']
    # Try Support Vector Machines
    if grid:
        parameters = {  'kernel':('poly', 'rbf', 'sigmoid'), \
                        'C':[1, 10, 100] }
                       # 'gamma':['auto', 0.01, 0.001] }
        print("\nSVM hyperparameter grid search for first wavelet components:")
        svc = SVC()
        clf = GridSearchCV(svc, cv=5, n_jobs=18, param_grid=parameters)
        clf.fit(my_X_train, my_y_train.ravel())
        print("-------------------------------------------------------------------")
        print("SVM Best params:")
        print(clf.best_params_)
        print("-------------------------------------------------------------------")
        print("SVM Best score:")
        print(clf.best_score_)
        print("-------------------------------------------------------------------")
        print("SVM All results:")
        print(clf.cv_results_)
        print("-------------------------------------------------------------------")
    elif cross_validation:
        clf = SVC(C=C, kernel=kernel, gamma=gamma)
        # 5-fold cross-validation
        scores = cross_val_score(clf, my_X_train, my_y_train.ravel(), cv=5, scoring="f1_weighted")
        print(scores)
        print("-------------------------------------------------------------------  ")
        print("Mean weighted F1-score: %f" % np.mean(scores))
        print("Std weighted F1-score: %f" % np.std(scores))
        print("-------------------------------------------------------------------  ")
    else:
        clf = SVC(C=C, kernel=kernel, gamma=gamma)
        print(clf)
        clf.fit(my_X_train, my_y_train.ravel())
        y_pred = clf.predict(my_X_test)

        # Store the classifier to reuse later
        pickle.dump(clf, open("%s/%s_SVM_clf.pickle" % (OUTPUT_PREFIX, test_file_prefix), 'wb'))

        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        if single_file:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[idx_test[i]])
                else:
                    true_positives.append(coordinates[idx_test[i]])
        else:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[i])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[i])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[i])
                else:
                    true_positives.append(coordinates[i])

        with open("%s/%s_SVM_false_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fp_file:
            for peak in false_positives:
                fp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_SVM_false_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fn_file:
            for peak in false_negatives:
                fn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_SVM_true_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tp_file:
            for peak in true_positives:
                tp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_SVM_true_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tn_file:
            for peak in true_negatives:
                tn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))

        print("-------------------------------------------------------------------")
        print("SVM Confusion Matrix:")
        print(confusion_matrix(my_y_test, y_pred))
        print("-------------------------------------------------------------------")
        print("F1-score (macro): %.3f" % f1_score(my_y_test, y_pred, average='macro'))
        print("F1-score (micro): %.3f" % f1_score(my_y_test, y_pred, average='micro'))
        print("F1-score (weighted): %.3f\n" % f1_score(my_y_test, y_pred, average='weighted'))
        print(classification_report(my_y_test, y_pred, target_names=target_names))
        print("-------------------------------------------------------------------")
        print("Accuracy: {}".format(accuracy_score(my_y_test, y_pred)))

#        # Plot ROC and report AUC
#        probas = clf.predict_proba(my_X_test)
#        fpr, tpr, thresholds = roc_curve(my_X_test, probas[:, 1])
#        print("AUC: {}".format(auc(fpr, tpr)))
#
#        plt.plot(fpr, tpr);
#        plt.title('SVM Receiver Operating Characteristic')
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.axis('equal')
#        plt.xlim((0, 1));
#        plt.ylim((0, 1));
#        plt.plot(range(2), range(2), '--');
#        plt.savefig("%s/%s_SVM_ROC.png" % (OUTPUT_PREFIX, test_file_prefix), dpi=600)


def classify_AdaBoost(grid, cross_validation, my_X_train, my_X_test, my_y_train, my_y_test, coordinates, n_estimators, algorithm, single_file, test_file_prefix):
    print('AdaBoost classifier called with args: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s' % (grid, cross_validation, my_X_train.shape, my_X_test.shape, my_y_train.shape, my_y_test.shape, len(coordinates), n_estimators, algorithm, single_file, test_file_prefix))
    target_names = ['non-functional binding', 'functional binding']
    # Try Support Vector Machines
    if grid:
        parameters = {  'algorithm':('SAMME', 'SAMME.R'), \
                        'n_estimators':[100, 200, 400, 600] }
        print("\nAdaBoost hyperparameter grid search for first wavelet components:")
        abc = AdaBoostClassifier()
        clf = GridSearchCV(abc, cv=5, n_jobs=18, param_grid=parameters)
        clf.fit(my_X_train, my_y_train.ravel())
        print("-------------------------------------------------------------------")
        print("AdaBoost Best params:")
        print(clf.best_params_)
        print("-------------------------------------------------------------------")
        print("AdaBoost Best score:")
        print(clf.best_score_)
        print("-------------------------------------------------------------------")
        print("AdaBoost All results:")
        print(clf.cv_results_)
        print("-------------------------------------------------------------------")
    elif cross_validation:
        clf = AdaBoostClassifier(n_estimators=n_estimators, algorithm=algorithm)
        # 5-fold cross-validation
        scores = cross_val_score(clf, my_X_train, my_y_train.ravel(), cv=5, scoring="f1_weighted")
        print(scores)
        print("-------------------------------------------------------------------  ")
        print("Mean weighted F1-score: %f" % np.mean(scores))
        print("Std weighted F1-score: %f" % np.std(scores))
        print("-------------------------------------------------------------------  ")
    else:
        clf = AdaBoostClassifier(n_estimators=n_estimators, algorithm=algorithm)
        print(clf)
        clf.fit(my_X_train, my_y_train.ravel())
        y_pred = clf.predict(my_X_test)

        # Store the classifier to reuse later
        pickle.dump(clf, open("%s/%s_AdaBoost_clf.pickle" % (OUTPUT_PREFIX, test_file_prefix), 'wb'))

        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        if single_file:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[idx_test[i]])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[idx_test[i]])
                else:
                    true_positives.append(coordinates[idx_test[i]])
        else:
            for i in range(len(y_pred)):
                if my_y_test[i] == 0 and y_pred[i] == 1:
                    false_positives.append(coordinates[i])
                elif my_y_test[i] == 1 and y_pred[i] == 0:
                    false_negatives.append(coordinates[i])
                elif my_y_test[i] == 0:
                    true_negatives.append(coordinates[i])
                else:
                    true_positives.append(coordinates[i])

        with open("%s/%s_AdaBoost_false_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fp_file:
            for peak in false_positives:
                fp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_AdaBoost_false_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as fn_file:
            for peak in false_negatives:
                fn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_AdaBoost_true_positives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tp_file:
            for peak in true_positives:
                tp_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))
        with open("%s/%s_AdaBoost_true_negatives.bed" % (OUTPUT_PREFIX, test_file_prefix), 'w') as tn_file:
            for peak in true_negatives:
                tn_file.write("%s\t%s\t%s\n" % (peak[0], peak[1], peak[2]))

        print("-------------------------------------------------------------------")
        print("AdaBoost Confusion Matrix:")
        print(confusion_matrix(my_y_test, y_pred))
        print("-------------------------------------------------------------------")
        print("F1-score (macro): %.3f" % f1_score(my_y_test, y_pred, average='macro'))
        print("F1-score (micro): %.3f" % f1_score(my_y_test, y_pred, average='micro'))
        print("F1-score (weighted): %.3f\n" % f1_score(my_y_test, y_pred, average='weighted'))
        print(classification_report(my_y_test, y_pred, target_names=target_names))
        print("-------------------------------------------------------------------")
        print("Accuracy: {}".format(accuracy_score(my_y_test, y_pred)))

#        # Plot ROC and report AUC
#        probas = clf.predict_proba(my_X_test)
#        fpr, tpr, thresholds = roc_curve(my_X_test, probas[:, 1])
#        print("AUC: {}".format(auc(fpr, tpr)))
#
#        plt.plot(fpr, tpr);
#        plt.title('AdaBoost Receiver Operating Characteristic')
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.axis('equal')
#        plt.xlim((0, 1));
#        plt.ylim((0, 1));
#        plt.plot(range(2), range(2), '--');
#        plt.savefig("%s/%s_AdaBoost_ROC.png" % (OUTPUT_PREFIX, test_file_prefix), dpi=600)

