import tensorflow as tensorflow
import keras as keras
from keras import regularizers
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from copy import copy
import numpy as np
import os
import sys
import json
import random
from data.load_rnn import load_pure_lstm
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, \
    normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct
import tensorflow as tf



window_len = 30
batch_size = 32
tra_date ='2007-01-03'
val_date = '2015-01-02'
tes_date = '2016-01-04'
split_dates = [tra_date, val_date, tes_date]
	# read data configure file
with open("D:/Internship/NExT/4EBaseMetal/exp/lstm_data.conf") as fin:
	fname_columns = json.load(fin)
print(fname_columns)
for fname in fname_columns:
	print('read columns:', fname_columns[fname], 'from:', fname)
		# time_series = read_single_csv(fname, fname_columns[fname])
		# print(time_series)
	# load data
X_tr, y_tr, X_val, y_val, X_tes, y_tes = load_pure_lstm(fname_columns, 'LMCADY', 'log_1d_return', split_dates, window_len, 3)
	#print(X_tr.shape[2])

model = Sequential()
model.add(LSTM(8, input_shape=(window_len,5), return_sequences=False,kernel_regularizer=regularizers.l2(0.01)))  # TODO: input_shape=(timesteps ,data_dim)
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None, beta_initializer="zero", gamma_initializer="one"))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=[metrics.binary_accuracy,metrics.mae])#[metrics.binary_accuracy])

hist = model.fit(X_tr,y_tr,batch_size=32,epochs=30, validation_data = (X_val, y_val))
#hist_dict = hist.history()
model.summary()
# predict = model.predict(X_tr,batch_size=1)
# print(predict)
score = model.evaluate(X_tes, y_tes, batch_size=32)
print(score)



import matplotlib.pyplot as plt

acc = hist.history['binary_accuracy']
val_acc = hist.history['val_binary_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')  # bo for blue dot 蓝色点
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clar figure


plt.plot(epochs, acc, 'bo', label='Training acc')  # bo for blue dot 蓝色点
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

