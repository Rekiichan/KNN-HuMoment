import pandas as pd
from statistical import test
from get_feature import get_feature, get_features
from write_data import write_file

if __name__ == "__main__":
    
    # get feature and write data feature
    
    suplo_feature, tomato_feature = get_features()
    write_file(suplo_feature,'data/suplo_data/suplo_feature')
    write_file(tomato_feature,'data/tomato_data/tomato_feature')
    
    # load data and thong ke
    
    file_path = 'data/train'
    train_data_suplo = pd.read_csv(
        f'{file_path}/train_data_suplo.csv').to_numpy()
    train_data_tomato = pd.read_csv(
        f'{file_path}/train_data_tomato.csv').to_numpy()
    test_data_suplo = pd.read_csv(
        f'{file_path}/test_data_suplo.csv').to_numpy()
    test_data_tomato = pd.read_csv(
        f'{file_path}/test_data_tomato.csv').to_numpy()
    print("\nvoi k = 3: ")
    RC, PR, ACC = test(train_data_suplo, train_data_tomato,
                       test_data_suplo, test_data_tomato, 3)
    print(f'RC: {RC}')
    print(f'PR: {PR}')
    print(f'ACC: {ACC}\n')
    print("voi k = 5: ")
    RC, PR, ACC = test(train_data_suplo, train_data_tomato,
                       test_data_suplo, test_data_tomato, 5)
    print(f'RC: {RC}')
    print(f'PR: {PR}')
    print(f'ACC: {ACC}')

