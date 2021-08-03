"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/3/2021
Homework #4 - Q1
Analyzing patient data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn



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

    print(df["fetal state"])
    print("hello world")


main()