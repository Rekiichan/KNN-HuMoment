import cv2
import numpy as np
import csv
import pandas as pd
from caculate_humoment import caculate_HuMM
from get_feature import get_features, get_feature
from write_data import write_file


def split_data(file_path):
    data = pd.read_csv(file_path)
    test_data = np.zeros((10, 7))
    train_data = np.zeros((50, 7))

    for i in range(0, 10):
        for j in range(0, 7):
            test_data[i][j] = data.loc[i][j]
    write_file(test_data, 'data/train/test_data')
# index = 0
# for key in range(0, 10, 2):
    for i in range(0, 10):
        for j in range(0, 7):
            # if i == key or i == key + 1:
            #     continue
            train_data[i][j] = data.loc[i][j]
            index += 1
            
    write_file(train_data, 'data/train/train_data')
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = split_data('data/suplo_data/suplo_feature.csv')

