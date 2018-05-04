from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Flatten
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint
import data_prepare as data
import numpy as np

tokens = ['etc', 'eth', 'lsk', 'ltc', 'str', 'strat', 'sys', 'xem', 'xmr', 'xrp']
time_stamp = 900
batch_size = 13
features = batch_size

def build_model(units):
    model = Sequential()
    model.add(LSTM(input_shape=(1, features), activation="tanh",
                   units=units, use_bias=False, dropout=0.2, return_sequences=False
                   ))
    # model.add(LSTM(input_shape=(1, features), activation="linear",
    #                units=int(units/2), use_bias=True, dropout=0.25, return_sequences=False))
    # model.add(Flatten())
    # model.add(Dense(units=(int(units / 3) * 2)))
    # model.add(Dense(units=(int(units / 3))))
    model.add(Dense(units=1))
    model.add(LeakyReLU())
    opt = Adam(lr=0.000007)
    model.compile(loss="mse", optimizer=opt)

    return model


neural = build_model(23)
filepath="models/network_timestamp"+str(time_stamp)+"_batch"+str(batch_size) + ".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto', period=10)
token = 'eth'
print('loading data for token: '+token)

train_data, answers = data.get_prepared_data(token, time_stamp, batch_size)
print('train data size: '+str(len(train_data)))
rest = len(train_data)%features
train_data = train_data[rest::]
answers = answers[rest::]
answers = np.reshape(answers, newshape=(len(answers), 1))
train_data = np.reshape(train_data, newshape=(int(len(train_data)),1 , features))
print(train_data.shape)
print(train_data[3])

history = neural.fit(train_data, answers, epochs=10000, batch_size=5700,
                     verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpoint])








