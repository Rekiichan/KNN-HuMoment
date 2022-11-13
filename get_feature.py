import cv2
import numpy as np
from caculate_humoment import caculate_HuMM

def get_feature(str):
    feature_matrix = np.zeros((10,7))
    for i in range(0,10):
        tmp = caculate_HuMM(f'{str} {i}.jpg')
        for j in range(0,7):
            feature_matrix[i][j] = tmp[j]
    return feature_matrix

def get_features():
    lst = []
    suplo_feature = get_feature('gray_img/suplo/suplo gray')
    tomato_feature = get_feature('gray_img/tomato/tomato gray')
    # print(suplo_feature)
    # print(tomato_feature)
    return suplo_feature, tomato_feature

if __name__ == "__main__":
    get_features()


