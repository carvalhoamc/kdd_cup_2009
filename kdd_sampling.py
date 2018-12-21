import numpy as np
import pandas as pd
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import os
import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import Isomap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV

from dialnd_imbalanced_algorithms.smote import SMOTEBoost

'''df1 = pd.read_csv('./folder/orange_large_train.data.chunk1',sep='\t')
col = df1.columns
df2 = pd.read_csv('./folder/orange_large_train.data.chunk2',sep='\t',header=None)
df3 = pd.read_csv('./folder/orange_large_train.data.chunk3',sep='\t',header=None)
df4 = pd.read_csv('./folder/orange_large_train.data.chunk4',sep='\t',header=None)
df5 = pd.read_csv('./folder/orange_large_train.data.chunk5',sep='\t',header=None)
frame = [df1,df2,df3,df4,df5]
df = pd.concat(frame)
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
print(df5.shape)
print(df.shape[0])
df_label_churn = pd.read_csv('./folder/orange_large_train_churn.labels')
df_label_appetency = pd.read_csv('./folder/orange_large_train_appetency.labels')
df_label_upselling = pd.read_csv('./folder/orange_large_train_upselling.labels')
print(df_label_churn.shape)
print(df_label_appetency.shape)
print(df_label_upselling.shape)
'''



output_dir = './output_dir/'
datasets = ['kdd_churn','kdd_appetency','kdd_upselling']
train_smote_ext = ["_UNDERSAMPLING","_train", "_SMOTE"]
nfolds = 2
base_estimator = AdaBoostClassifier(n_estimators=15)

classifiers = {
    "RF": RandomForestClassifier(n_estimators=50,n_jobs=-1),
    "KNN": KNeighborsClassifier(),
    "DTREE": DecisionTreeClassifier(),
    "GNB": GaussianNB(),
    "LRG": LogisticRegression(),
    "ABC": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=500,hidden_layer_sizes=(300,30)),
    "KDA": QuadraticDiscriminantAnalysis(),
    "SVM": SVC(probability=True),
    "SGD": SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
}


def createValidationData(folder):
    """
    Create sub datasets for cross validation purpose
    :param datasets: List of datasets
    :param folder: Where datasets was stored
    :return:
    """

    for dataset in datasets:
        print(dataset)
        fname = os.path.join(folder, ''.join([dataset, ".csv"]))
        data = pd.read_csv(fname)
        data = data.values
        X = normalize(data[:,:-1])
        Y = np.array(data[:,data.shape[1]-1])
        Y = Y.reshape(len(Y),1)
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True)

        for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
            train = pd.DataFrame(np.hstack((X_train, y_train)))
            test = pd.DataFrame(np.hstack((X_test, y_test)))
            os.makedirs(os.path.join(folder, dataset, str(fold)))
            train.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"])), header=False,
                         index=False)
            test.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"])), header=False,
                        index=False)


def runSMOTEvariationsGen(folder):
    """
    Create files with SMOTE preprocessing and without preprocessing.
    :param datasets: datasets.
    :param folder:   cross-validation folders.
    :return:
    """
    smote = SMOTE(k_neighbors=10,n_jobs=-1)
    undersampler = InstanceHardnessThreshold(random_state=0, estimator=LogisticRegression(solver='lbfgs'))

    for dataset in datasets:
        for fold in range(nfolds):
            path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"]))
            train = np.genfromtxt(path, delimiter=',')
            X = train[:, 0:train.shape[1] - 1]
            Y = train[:, train.shape[1] - 1]


            #Undersampling
            print("Undersampling..." + dataset)
            print(fold)
            X_res, y_res = undersampler.fit_resample(X, Y)
            y_res = y_res.reshape(len(y_res), 1)
            newdata = np.hstack([X_res, y_res])
            newtrain = pd.DataFrame(np.vstack([train, newdata]))
            newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_UNDERSAMPLING.csv"])),
                            header=False, index=False)

            # SMOTE
            print("SMOTE..." + dataset)
            print(fold)
            X_res, y_res = smote.fit_sample(X, Y)
            y_res = y_res.reshape(len(y_res), 1)
            newdata = np.hstack([X_res, y_res])
            newtrain = pd.DataFrame(np.vstack([train, newdata]))
            newtrain.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_SMOTE.csv"])),
                            header=False, index=False)


def converteY(Y):
    '''
    Nos novos datasets, Y != [-1,1], precisamos converter para [-1,1]
    :param Y:
    :return:
    '''
    c = np.unique(Y)
    if not np.all(c == [-1, 1]):
        Y[Y == 1] = -1
        Y[Y == 2] = 1
    return Y


def runClassification(folder, SMOTE=False):
    dfcol = ['ID', 'DATASET', 'FOLD', 'PREPROC', 'ALGORITHM', 'MODE', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA', 'AUC']
    df = pd.DataFrame(columns=dfcol)
    i = 0

    for dataset in datasets:
        for fold in range(nfolds):
            test_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"]))
            test = np.genfromtxt(test_path, delimiter=',')
            X_test = test[:, 0:test.shape[1] - 1]
            Y_test = test[:, test.shape[1] - 1]
            Y_test = converteY(Y_test)

            # SMOTE LIKE CLASSIFICATION
            if SMOTE == True:
                for ext in train_smote_ext:
                    train_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, ext, ".csv"]))
                    train = np.genfromtxt(train_path, delimiter=',')
                    X_train = train[:, 0:train.shape[1] - 1]
                    Y_train = train[:, train.shape[1] - 1]
                    Y_train = converteY(Y_train)# nao precisa para multiclasse

                    if ext == "_train":
                        X, Y = X_train, Y_train  # original dataset for plotting
                    for name, clf in classifiers.items():
                        #clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
                        clf.fit(X_train, Y_train)
                        Y_pred = clf.predict(X_test)
                        res = classification_report_imbalanced(Y_test, Y_pred)
                        identificador = dataset + '_' + ext + '_' + name
                        aux = res.split()
                        score = aux[-7:-1]
                        df.at[i, 'ID'] = identificador
                        df.at[i, 'DATASET'] = dataset
                        df.at[i, 'FOLD'] = fold
                        df.at[i, 'PREPROC'] = ext
                        df.at[i, 'ALGORITHM'] = name
                        df.at[i, 'MODE'] = 'NC'  # No Compression
                        df.at[i, 'PRE'] = score[0]
                        df.at[i, 'REC'] = score[1]
                        df.at[i, 'SPE'] = score[2]
                        df.at[i, 'F1'] = score[3]
                        df.at[i, 'GEO'] = score[4]
                        df.at[i, 'IBA'] = score[5]
                        df.at[i, 'AUC'] = roc_auc_score(Y_test, Y_pred) #binario
                        #df.at[i, 'AUC'] = -1  # multiclasse

                        i = i + 1
                        print(fold, identificador)
                        print(df)
            df.to_csv('./output_dir/results.csv', index=False)

        #df.to_csv('results.csv', index=False)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))


def main():
    start = time.time()
    folder_experiments = './dados/'
    le = preprocessing.LabelEncoder()

    df_train = pd.read_csv('./folder/orange_small_train.data', sep='\t')
    df_train.fillna(0, inplace=True)  # clear NaN
    df_test = pd.read_csv('./folder/orange_small_test.data', sep='\t')
    df_test.fillna(0, inplace=True)  # clear NaN
    df_label_appetency = pd.read_csv('./folder/orange_small_train_appetency.labels', sep='\t', header=None)
    df_label_churn = pd.read_csv('./folder/orange_small_train_churn.labels', sep='\t', header=None)
    df_label_upselling = pd.read_csv('./folder/orange_small_train_upselling.labels', sep='\t', header=None)

    # identify categorical columns
    cols = df_train.columns
    num_cols = df_train._get_numeric_data().columns
    categorical_cols = list(set(cols) - set(num_cols))
    for c in categorical_cols:
        v = list(df_train[c])
        new_v = le.fit_transform(v)
        df_train[c] = new_v  # new values to col dataframe

    np_train = df_train.values
    np_appetency =  df_label_appetency.values
    np_churn = df_label_churn.values
    np_upselling = df_label_upselling.values


    #feature selection
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(np_train, np_appetency)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(np_train)
    np_train = X_new
    df_train_appetency = pd.DataFrame(np.hstack((np_train, np_appetency)))

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(np_train, np_churn)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(np_train)
    np_train = X_new
    df_train_churn = pd.DataFrame(np.hstack((np_train, np_churn)))

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(np_train, np_upselling)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(np_train)
    np_train = X_new
    df_train_upselling = pd.DataFrame(np.hstack((np_train, np_upselling)))

    df_train_appetency.to_csv('./dados/kdd_appetency.csv')
    df_train_churn.to_csv('./dados/kdd_churn.csv')
    df_train_upselling.to_csv('./dados/kdd_upselling.csv')


    createValidationData(folder_experiments)
    runSMOTEvariationsGen(folder_experiments)
    runClassification(folder_experiments,SMOTE=True)

    end = time.time()
    print("Total Execution Time : ")
    timer(start, end)


if __name__ == "__main__":
    main()
