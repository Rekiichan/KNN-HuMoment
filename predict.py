import cv2
import math
import numpy as np
import csv
import pandas as pd
from caculate_humoment import caculate_HuMM
from get_feature import get_features, get_feature
from process_data import split_data

def predict(train_data_tomato, train_data_suplo, test_data, K_test=3):
    distance_array_tomato = np.zeros((8))
    distance_array_suplo = np.zeros((8))
    
    for row in range (8):
        for col in range (7):
            distance_array_tomato[row] += (train_data_tomato[row][col] - test_data[col])**2
        distance_array_tomato[row] = math.sqrt(distance_array_tomato[row])

    for row in range (8):
        for col in range (7):
            distance_array_suplo[row] += (train_data_suplo[row][col] - test_data[col])**2
        distance_array_suplo[row] = math.sqrt(distance_array_suplo[row])
        
    distance_array_suplo = np.sort(distance_array_suplo,kind='heapsort') 
    distance_array_tomato = np.sort(distance_array_tomato,kind='heapsort') 
    
    Tomato_count = 0
    Suplo_count = 0
    while (Tomato_count + Suplo_count) < K_test:
        if (distance_array_tomato[Tomato_count] > distance_array_suplo[Suplo_count]):
            Suplo_count += 1
        else:
            Tomato_count += 1
            
    if Suplo_count > Tomato_count:
        return "sup lo"
    else:
        return "ca chua"
    
if __name__ == "__main__":
    file_path_suplo = 'data/suplo_data/suplo_feature.csv'
    file_path_tomato = 'data/tomato_data/tomato_feature.csv'
    train_data_tomato, test_data_tomato = split_data(file_path_tomato)
    train_data_suplo, test_data_suplo = split_data(file_path_suplo)
    
    res = predict(train_data_tomato, train_data_suplo, test_data_tomato[1], 5)
    print(res)