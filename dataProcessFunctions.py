import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



csvPath = "csvFile/basketballCU.csv"


def decssionTree(mssk_result):
    return 0



if __name__ == '__main__':
    mssk_result = pd.DataFrame
    mssk_result = pd.read_csv(csvPath, index_col=0)

    x = mssk_result['mean'].tolist()
    y = mssk_result['Lap_Mean'].tolist()
    plt.scatter(x, y)
    plt.show()
    # mssk_array  = mssk_result.to_numpy()
    # features = mssk_result.columns
    # feature_list = features.tolist()
    # print(feature_list)