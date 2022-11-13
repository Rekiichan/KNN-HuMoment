import cv2
import numpy as np
import csv
import pandas as pd
from caculate_humoment import caculate_HuMM
from get_feature import get_features,get_feature

def write_file(features, file_path):
    # suplo_features, tomato_features = get_features()
    header = ['S1','S2','S3','S4','S5','S6','S7']
    with open(f'{file_path}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(0,10):
            writer.writerow(features[i])
        f.close()

def write_data_train(data_train, data_test, file_path):
    header = ['S1','S2','S3','S4','S5','S6','S7']
    write_file(data_test, f'data/train/test_data_{file_path}')
    
    with open(f'data/train/train_data_{file_path}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(0,40):
            writer.writerow(data_train[i])
        f.close()

if __name__ == "__main__":
    write_file()

