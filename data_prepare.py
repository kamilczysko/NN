import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sk = MinMaxScaler(feature_range=(0,1))

def get_close_prices(file):
    csv_file = pandas.read_csv('marketData/'+file+'.csv', delimiter=';')
    raw_data = csv_file['close'].values
    raw_data = np.reshape(raw_data, newshape=(len(raw_data), 1))
    dataset = sk.fit_transform(raw_data)
    print("raw data size: "+str(len(raw_data)))
    return dataset

def seperate_data(array, training_size):

    arr = array[training_size:]
    separated_data = []
    answers = []

    for i in range(len(arr)-training_size):
        separated_data.append(arr[:training_size])
        answers.append(arr[training_size])
        arr = arr[1:]

    print('after split data: '+str(len(separated_data)))
    return separated_data, answers

def get_prepared_data(token, time_stamp, batch_size):
    file = token.upper()+'_'+str(time_stamp)
    raw_data = get_close_prices(file)
    separated_data = seperate_data(raw_data, batch_size)

    return separated_data

# pr = get_close_prices('etc_300')
# a,b = seperate_data(pr , 9)
# print(a[:6])
# print(b[:6])
