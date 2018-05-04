from keras.models import load_model
import data_prepare as data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sk = MinMaxScaler(feature_range=(0,1))


token = 'eth'
timestamp = 14400
batch_size = 8
print('loading model')
# model = load_model('models/network_timestamp'+str(timestamp)+'_batch'+str(batch_size)+'.h5')
model = load_model('models/network_timestamp900_batch'+str(batch_size)+'.h5')
print('loading test data')
raw_data = data.get_close_prices(token+"_"+str(timestamp))
print('loading separaterd data')
data_to_process,_ = data.seperate_data(raw_data, batch_size) #raw_data[:features]#np.reshape(raw_data[:batch_size], newshape=(1, 1, batch_size))
result_array = raw_data[:batch_size]
print('testing')
for i in data_to_process:#range(int(len(raw_data)/batch_size)):
    t = np.reshape(i, newshape=(1, 1, batch_size))
    print('predicting')
    result_from_neural = model.predict(t, batch_size)
    # print(str(result_from_neural))
    result_array = np.append(result_array, result_from_neural)

# raw_data = sk.inverse_transform(raw_data)
# result_array = sk.inverse_transform(result_array)

plt.plot(raw_data[:400], "r")

plt.plot(result_array[:400], "b.")
plt.show()
