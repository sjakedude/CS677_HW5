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
from sklearn import metrics


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

    print(accuracy)


main()
