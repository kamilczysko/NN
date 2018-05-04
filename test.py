import numpy as np
from sklearn.preprocessing import MinMaxScaler
sk = MinMaxScaler(feature_range=(0,1))
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float64)
a = np.reshape(a, newshape=(len(a), 1))

b = np.array([11,22,33,44,55,66,77,88,99])
print(a[-2::])


