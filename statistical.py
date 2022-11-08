import cv2
import numpy as np
import csv
import pandas as pd
from caculate_humoment import caculate_HuMM
from get_feature import get_features,get_feature
from process_data import split_data
from predict import predict

#ca chua N, sup lo P
def test(train_data_suplo,train_data_tomato,test_data_suplo,test_data_tomato,K_test = 3):
    RC  = 0.0
    PR  = 0.0
    ACC = 0.0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    
    for i in range(0,5):
        tmp = predict(train_data_tomato[i*8:i*8+8], train_data_suplo[i*8:i*8+8], test_data_suplo[i*2], K_test)
        if tmp == 'sup lo':
            TP += 1
        else:
            FP += 1
        tmp = predict(train_data_tomato[i*8:i*8+8], train_data_suplo[i*8:i*8+8], test_data_suplo[i*2+1], K_test)
        if tmp == 'sup lo':
            TP += 1
        else:
            FP += 1
            
        tmp = predict(train_data_tomato[i*8:i*8+8], train_data_suplo[i*8:i*8+8], test_data_tomato[i*2], K_test)
        if tmp == 'ca chua':
            TN += 1
        else:
            FN += 1
        tmp = predict(train_data_tomato[i*8:i*8+8], train_data_suplo[i*8:i*8+8], test_data_tomato[i*2+1], K_test)
        if tmp == 'ca chua':
            TN += 1
        else:
            FN += 1

    print(f'TP: {TP}')
    print(f'TN: {TN}')
    print(f'FN: {FN}')
    print(f'FP: {FP}')
    print('-----------')
    
    RC = TP / (TP + FN)
    PR = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    return RC, PR, ACC
    
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