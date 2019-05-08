#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import itertools as itt
from sklearn import tree
from subprocess import check_call

# Function that cleans the Data Set


def CleanData(data_full):
    data_full = data_full.replace("\t", "")
    data_full = data_full.replace("ckd\t", "ckd")
    data_full = data_full.replace("notckd\t", "notckd")
    data_full = data_full.replace("\tckd", "ckd")
    data_full = data_full.replace("\tnotckd", "notckd")
    data_full = data_full.replace("yes\t", "yes")
    data_full = data_full.replace("no\t", "no")
    data_full = data_full.replace("\tyes", "yes")
    data_full = data_full.replace("\tno", "no")
    data_full = data_full.replace(",,", ",")
    data_full = data_full.replace(" yes", "yes")
    data_full = data_full.replace(key_list, key_value)
    return data_full

# Function that normalizes the matrix


def Normalization(matr, mean, std):
    (rows, cols) = matr.shape
    matr_norm = np.zeros((len(matr[:, 1]), cols), dtype=float)
    for i in range(0, cols):
        for j in range(0, rows):
            matr_norm[j, i] = (matr[j, i] - mean[i, 0])/std[i, 0]
    return matr_norm

# Function that computes the Ridge Regression


def RidgeRegression(index_col, X_train, y_train, X_test, mean, std):
    n_r, n_c = np.shape(X_train)
    mean_f = mean[index_col, 0]
    std_f = std[index_col, 0]

    I = np.eye(n_c)
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_train), X_train)+10*I), np.transpose(X_train)), y_train)
    y_hat_norm = np.dot(X_test, w)
    y_hat = y_hat_norm*std_f+mean_f
    return y_hat

# Function that rounds the values of the Data Set according to their possible ranges


def RoundData(data_final):
    al = data_final.columns.get_loc("Albumin")
    su = data_final.columns.get_loc("Sugar")
    sg = data_final.columns.get_loc("Specific Gravity")
    for i_r in range(len(data_final)):
        if data_final.iat[i_r, al] < 0:
            data_final.iat[i_r, al] = 0
        elif data_final.iat[i_r, al] > 5:
            data_final.iat[i_r, al] = 5
    for i_r in range(len(data_final)):
        if data_final.iat[i_r, su] < 0:
            data_final.iat[i_r, su] = 0
        elif data_final.iat[i_r, su] > 5:
            data_final.iat[i_r, su] = 5
    for i_r in range(len(data_final)):
        if data_final.iat[i_r, sg] < 1.0075:
            data_final.iat[i_r, sg] = 1.005
        elif data_final.iat[i_r, sg] > 1.0225:
            data_final.iat[i_r, sg] = 1.025
        elif data_final.iat[i_r, sg] >= 1.075 and data_final.iat[i_r, sg] < 1.0125:
            data_final.iat[i_r, sg] = 1.010
        elif data_final.iat[i_r, sg] >= 1.0125 and data_final.iat[i_r, sg] < 1.0175:
            data_final.iat[i_r, sg] = 1.015
        elif data_final.iat[i_r, sg] >= 1.0175 and data_final.iat[i_r, sg] < 1.0225:
            data_final.iat[i_r, sg] = 1.020
    data_final = data_final.round({"Age": 0, "Blood Pressure": 0, "Albumin": 0, "Sugar": 0,
                                   "Red Blood Cells": 0, "Pus Cell": 0, "Pus Cell Clumps": 0, "Bacteria": 0,
                                   "Blood Glucose Random": 0, "Blood Urea": 0,
                                   "Sodium": 0, "Packed Cell Volume": 0,
                                   "White Blood Cell Count": 0, "Hypertension": 0,
                                   "Diabetes Mellitus": 0, "Coronary Artery Disease": 0, "Appetite": 0,
                                   "Pedal Edema": 0, "Anemia": 0, "Class": 0})
    data_final = data_final.round({"Serum Creatinine": 1, "Potassium": 1, "Hemoglobin": 1, "Red Blood Cell Count": 1})

    return data_final


# Definition of the names of the features
feat_names = ["Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
              "Red Blood Cells", "Pus Cell", "Pus Cell Clumps", "Bacteria",
              "Blood Glucose Random", "Blood Urea", "Serum Creatinine",
              "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume",
              "White Blood Cell Count", "Red Blood Cell Count", "Hypertension",
              "Diabetes Mellitus", "Coronary Artery Disease", "Appetite",
              "Pedal Edema", "Anemia", "Class"]
# Definition of the letteral description
key_list = ["normal", "abnormal", "present", "notpresent", "yes", "no", "poor", "good",
            "ckd", "notckd"]
# Conversion in binary of the letteral description
key_value = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]

# Reading the Data Set
data_full = pd.read_csv("./dataset/chronic_kidney_disease.cvs", sep=',', skiprows=29, na_values=['?', '\t?'], names=feat_names)

# Cleaning the Data Set
data_full = CleanData(data_full)

with open('./file_full.csv', "w") as f:
    data_full.to_csv(f)

data_20 = data_full.dropna(thresh=20)
with open('./file_no_na_20.csv', "w") as f:
    data_20.to_csv(f)

data_25 = data_20.dropna(thresh=25)
with open('./file_no_na_25.csv', "w") as f:
    data_25.to_csv(f)

X = data_25.values
(rows, cols) = X.shape
X_mean = np.mean(X, 0, dtype=float)
X_mean = X_mean.reshape(cols, 1)
X_std = np.std(X, 0, dtype=float)
X_std = X_std.reshape(cols, 1)
X_norm = Normalization(X, X_mean, X_std)
(r, c) = np.shape(X_norm)


for val in range(0, 5):
    print(val)

    data_20 = data_full.dropna(thresh=20)
    with open('./file_no_na_20.csv', "w") as f:
        data_20.to_csv(f)

    data_24 = data_20.dropna(thresh=24-val)
    with open('./file_24.csv', 'w') as f:
        data_24.to_csv(f)

    idrow, idcol = np.where(pd.isnull(data_24))
    index = np.column_stack([data_24.index[idrow], data_24.columns[idcol]])

    cat = []
    pat = index[0, 0]
    ind_new = []
    for p in range(0, len(index)):
        if p < len(index)-1 and index[p, 0] != index[p+1, 0]:
            cat.append(index[p, 1])
            cat = np.asarray(cat)
            cat.reshape(1, len(cat))
            ind_new.append(np.hstack([np.array(pat), cat]))
            cat = []
            pat = index[p+1, 0]
        elif p < len(index)-1 and index[p, 0] == index[p+1, 0]:
            cat.append(index[p, 1])
        elif p == len(index)-1:
            cat.append(index[p, 1])
            cat = np.asarray(cat)
            cat.reshape(1, len(cat))
            ind_new.append(np.hstack([np.array(pat), cat]))

    ind_mat = []
    ind_sh = np.array(ind_new).shape
    for item in ind_new:
        ind_mat = np.append(ind_mat, item)
    ind_mat = ind_mat.reshape(1, len(ind_mat))
    ind_mat = ind_mat.reshape(len(ind_new), -1)

    if val > 0:

        col_comb = list(itt.combinations(feat_names, val+1))
        col_mat = np.asmatrix(col_comb)
    else:

        col_comb = list(feat_names)
        col_mat = np.asarray(col_comb)
        dim_c = np.shape(col_mat)
        col_mat = col_mat.reshape(dim_c[0], 1)

    for c in range(len(col_mat)):
        test_a = pd.DataFrame()
        test = pd.DataFrame()
        pat_row = []

        for i in range(len(ind_mat)):

            if (val == 0 and np.array_equal(ind_mat[i, 1:val+2], col_mat[c, :]) is True) or (val > 0 and np.array_equal([ind_mat[i, 1:val+2]], col_mat[c, :]) is True):

                idx = int(ind_mat[i, 0])

                pat_row = np.append(pat_row, ind_mat[i, 0])

                line = data_24.loc[idx, :].copy()

                test_a = test_a.append(line)
        if not test_a.empty:
            test = test_a[feat_names]

            test_mat = test.values

            test_norm = Normalization(test_mat, X_mean, X_std)

            index_col = np.zeros(val+1)
            for idc in range(0, val+1):
                index_col[idc] = data_25.columns.get_loc(col_mat[c, idc])
            index_col = np.asarray(index_col, dtype=int)

            X_test = np.delete(test_norm, index_col, 1)

            X_train = np.delete(X_norm, index_col, 1)

            y_train = X_norm[:, index_col].reshape(r, val+1)

            for ind in range(0, val+1):
                y_hat = RidgeRegression(index_col[ind], X_train, y_train[:, ind], X_test, X_mean, X_std)
                s = data_full[col_mat[c, ind]]
                for i_pat in range(0, len(pat_row)):
                    s[int(pat_row[i_pat])] = y_hat[i_pat]

            with open('./file_full.csv', "w") as f:
                data_full.to_csv(f)

data_final = data_full.dropna(thresh=20)
data_final = RoundData(data_final)
with open('./file_final.csv', "w") as f:
    data_final.to_csv(f)


# Generate tree.
filetree = ["tree1", "tree2", "tree3"]
randstate = [1, 2, 6]
for o, k in enumerate(randstate):
    data_tree = data_final.drop(["Class"], axis=1)
    target = data_final["Class"].copy()
    feat_names_t = feat_names[0:24]
    target_names = ["Not CKD", "CKD"]
    clf = tree.DecisionTreeClassifier("entropy", random_state=k)
    clf = clf.fit(data_tree, target)
    dot_data = tree.export_graphviz(clf, out_file="Tree.dot",
                                    feature_names=feat_names_t,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    print("Features importance for random_state = %d:" % k)
    f = list(zip(feat_names, clf.feature_importances_))

    for x in f:
        print(str(x)[1:-1])

    print("")

    check_call(['dot', '-Tpdf', './Tree.dot', '-o', '%d_Tree.pdf' % o])


print("END")
