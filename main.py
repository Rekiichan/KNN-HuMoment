import cv2
import numpy as np
import csv
import pandas as pd
from caculate_humoment import caculate_HuMM
from get_feature import get_features,get_feature
from process_data import split_data
from predict import predict
from statistical import test

if __name__ == "__main__":
    file_path = 'data/train'
    train_data_suplo  = pd.read_csv(f'{file_path}/train_data_suplo.csv').to_numpy()
    train_data_tomato = pd.read_csv(f'{file_path}/train_data_tomato.csv').to_numpy()
    test_data_suplo   = pd.read_csv(f'{file_path}/test_data_suplo.csv').to_numpy()
    test_data_tomato  = pd.read_csv(f'{file_path}/test_data_tomato.csv').to_numpy()
    print("\nvoi k = 3: ")
    RC, PR, ACC  = test(train_data_suplo,train_data_tomato,test_data_suplo,test_data_tomato,3)
    print(f'RC: {RC}')
    print(f'PR: {PR}')
    print(f'ACC: {ACC}\n')
    print("voi k = 5: ")
    RC, PR, ACC  = test(train_data_suplo,train_data_tomato,test_data_suplo,test_data_tomato,5)
    print(f'RC: {RC}')
    print(f'PR: {PR}')
    print(f'ACC: {ACC}')
    # res = predict(train_data_tomato, train_data_suplo, test_data_suplo[0], 5)
    # print(res)