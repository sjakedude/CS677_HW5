"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/3/2021
Homework #5
Analyzing patient data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sn
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Adding Color column based on class
def calculate_fetal_state(row):
    if row["NSP"] == 1:
        return "N"
    else:
        return "A"


def main():

    # ==================
    # Question #1
    # ==================

    # Reading in dataframe
    df = pd.ExcelFile("data/CTG.xls").parse(2).drop(0).reset_index()

    # Calculating the fetal state
    df["fetal state"] = df.apply(lambda row: calculate_fetal_state(row), axis=1)

    # ==================
    # Question #2
    # ==================
    print("======================")
    print("Question #2")
    print("======================")

    # Separating into x and y
    x = df[["LB", "MLTV", "Width", "Variance"]]
    y = df["fetal state"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    NB_classifier = MultinomialNB().fit(x_train, y_train)
    y_predict = NB_classifier.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_predict)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_predict))

    print("\nAccuracy: " + str(round((accuracy * 100), 2)) + "%")

    acc_nb = accuracy
    index = 0
    tp_nb = 0
    fp_nb = 0
    tn_nb = 0
    fn_nb = 0
    y_test = y_test.values.tolist()

    for item in y_predict:
        if item == "N":
            if item == y_test[index]:
                tp_nb += 1
            else:
                fp_nb += 1
        else:
            if item == y_test[index]:
                tn_nb += 1
            else:
                fn_nb += 1
        index += 1
    tpr_nb = tp_nb / (tp_nb + fn_nb)
    tnr_nb = tn_nb / (tn_nb + fp_nb)

    # ==================
    # Question #3
    # ==================
    print("\n======================")
    print("Question #3")
    print("======================")

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_predict)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_predict))

    print("\nAccuracy: " + str(round((accuracy * 100), 2)) + "%")

    acc_dt = accuracy
    index = 0
    tp_dt = 0
    fp_dt = 0
    tn_dt = 0
    fn_dt = 0

    for item in y_predict:
        if item == "N":
            if item == y_test[index]:
                tp_dt += 1
            else:
                fp_dt += 1
        else:
            if item == y_test[index]:
                tn_dt += 1
            else:
                fn_dt += 1
        index += 1
    tpr_dt = tp_dt / (tp_dt + fn_dt)
    tnr_dt = tn_dt / (tn_dt + fp_dt)

    # ==================
    # Question #4
    # ==================
    print("\n======================")
    print("Question #4")
    print("======================")

    depths = [1, 2, 3, 4, 5]
    n_estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_accuracy = 0
    best_depth = None
    best_n = None

    for d in depths:
        for n in n_estimators:
            tree = RandomForestClassifier(
                criterion="entropy", max_depth=d, n_estimators=n, random_state=1
            )
            tree.fit(x_train, y_train)
            y_predict = tree.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, y_predict)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_depth = d
                best_n = n
    print("\nAccuracy: " + str(round((best_accuracy * 100), 2)) + "%")
    print("N=" + str(best_n))
    print("D=" + str(best_depth))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_predict))

    acc_rf = accuracy
    index = 0
    tp_rf = 0
    fp_rf = 0
    tn_rf = 0
    fn_rf = 0

    for item in y_predict:
        if item == "N":
            if item == y_test[index]:
                tp_rf += 1
            else:
                fp_rf += 1
        else:
            if item == y_test[index]:
                tn_rf += 1
            else:
                fn_rf += 1
        index += 1
    tpr_rf = tp_rf / (tp_rf + fn_rf)
    tnr_rf = tn_rf / (tn_rf + fp_rf)

    # ==================
    # Question #5
    # ==================
    print("\n======================")
    print("Question #5")
    print("======================")

    print("Table:\n")
    accuracy_table = pd.DataFrame(
        {
            "tp": [tp_nb, tp_dt, tp_rf],
            "fp": [fp_nb, fp_dt, fp_rf],
            "tn": [tn_nb, tn_dt, tn_rf],
            "fn": [fn_nb, fn_dt, fn_rf],
            "accuracy": [acc_nb, acc_dt, acc_rf],
            "tpr": [tpr_nb, tpr_dt, tpr_rf],
            "tnr": [tnr_nb, tnr_dt, tnr_rf],
        }
    )
    print(accuracy_table)


main()
