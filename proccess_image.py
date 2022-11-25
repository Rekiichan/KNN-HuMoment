import cv2 as cv
import numpy as np

def pre_processImage(img):
    rows, cols, highs = img.shape
    for r in range (rows):
        for c in range (cols):
            if img[r][c][0] > 100:
                img[r][c] = 1
            else:
                img[r][c] = 0
    return img
                
def processConvert(src):
    converted_image = np.zeros((src.shape))
    converted_image = 255 - src
    rows,cols = converted_image.shape
    for r in range (rows):
        for c in range (cols):
            if converted_image[r][c] > 30:
                converted_image[r][c] = 255
    converted_image = pre_processImage(converted_image)
    return converted_image

def cvtToGray(src):
    gray_img = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    return gray_img

def get_data(tem):
    S = []
    for i in range(10):
        S.append(tem+f" ({i}).jpg")
    return S
    
if __name__ == "__main__":
    S = get_data("tomato")
    tomato_data=[]
    for i in range(10):
        img = cv.imread(S[i])
        tomato_data.append(cvtToGray(img))
        tomato_data[i] = processConvert(tomato_data[i])
        cv.imwrite(f"tomato gray {i}.jpg",tomato_data[i])
        

    
    