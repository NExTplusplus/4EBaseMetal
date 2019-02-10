## Keras for deep learning
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, ConvLSTM2D, Activation
import keras
import sys
## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error

#for logging
import time

##matrix math
import numpy as np
import math

##plotting
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
##data processing
import pandas as pd
np.random.seed(7)


def bilstm_model(hidden_units, window_size, activation_function, loss_function, optimizer):
    """
    Initializes and creates the model to be used
    
    Arguments:
    window_size -- An integer that represents how many days of X_values the model can look at at once
    dropout_value -- A decimal representing how much dropout should be incorporated at each level, in this case 0.2
    activation_function -- A string to define the activation_function, in this case it is linear
    loss_function -- A string to define the loss function to be used, in the case it is mean squared error
    optimizer -- A string to define the optimizer to be used, in the case it is adam
    
    Returns:
    model -- A layer RNN that uses activation_function as its activation
             function, loss_function as its loss function, and optimizer as its optimizer
    """
    #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=False), input_shape=(window_size, X_train.shape[-1]),))
    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)
    
    return model
def lstm_model(hidden_units, window_size, activation_function, loss_function, optimizer):
     #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(LSTM(hidden_units, input_shape=(window_size, 5), activation='relu'))
    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)
    
    return model

def dense_model(hidden_units, window_size, activation_function, loss_function, optimizer):
     #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(Dense(hidden_units, input_dim=window_size*X_train.shape[-1]))
    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)
    
    return model

def cnn_lstm_model(hidden_units_cnn, hidden_units_lstm, window_size, activation_function, loss_function, optimizer):
    
    
    model = Sequential()
    model.add(Conv1D(filters=hidden_units_cnn, kernel_size=3, activation='relu', input_shape=(window_size, X_train.shape[-1])))
    model.add(Conv1D(filters=hidden_units_cnn, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(hidden_units_lstm, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation = activation_function)))

    # model = Sequential()
    # model.add(Conv2D(hidden_units_cnn,(2,2),strides=(1,1), activation= 'relu', padding = 'same', input_shape = (window_size, 5 , 1) ))
    # model.add(MaxPooling2D(pool_size=(2,2), data_format = 'channels_last'))
    # model.add(Flatten())
    # # model.add(TimeDistributed(Conv2D(hidden_units_cnn,(3,3),strides=(1,1), activation= 'relu', input_shape = (window_size, 5 , 1) )))
    # # model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
    # # model.add(TimeDistributed(Flatten()))
    # #model.add(Dropout(0.2))
    # model.add(LSTM(hidden_units_lstm))
    # #model.add(Dense(128, activation= 'relu'))
    # #Set activation function
    # model.add(Dense(1))
    #model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)

    return model

def convlstm_model(hidden_units_cnn, hidden_units_lstm, hidden_units_dense, window_size, activation_function, loss_function, optimizer):
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=hidden_units_cnn, kernel_size=(1,3), activation='relu', input_shape=(2, 1, 5, 5)))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(hidden_units_lstm, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(hidden_units_dense, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss = loss_function, optimizer = optimizer)

    return model


def fit_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_num, num_epoch, hidden_units_lstm, window_size):
    """
    Fits the model to the training data
    
    Arguments:
    model -- The previously initalized 3 layer Recurrent Neural Network
    X_train -- A tensor of shape (N, S, F) that represents the x values of the training data
    Y_train -- A tensor of shape (N,) that represents the y values of the training data
    batch_num -- An integer representing the batch size to be used, in this case 1024
    num_epoch -- An integer defining the number of epochs to be run, in this case 100
    val_split -- A decimal representing the proportion of training data to be used as validation data
    
    Returns:
    model -- The 3 layer Recurrent Neural Network that has been fitted to the training data
    training_time -- An integer representing the amount of time (in seconds) that the model was training
    """
    #Record the time the model starts training
    class testHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.test_loss = []
        def on_epoch_end(self, epoch, logs={}):
            score1 = model.evaluate(X_test, Y_test)
            print('test loss:',score1)
            self.test_loss.append(score1)
    start = time.time()

    #Train the model on X_train and Y_train
    test_content = testHistory()
    hist = model.fit(X_train, Y_train, batch_size= batch_num, epochs=num_epoch, validation_data = (X_val, Y_val),callbacks=[test_content])
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    test_loss = test_content.test_loss
    epochs = len(loss) + 1
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Loss')
    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.plot(test_loss, 'y', label='Testing loss')
    ax.legend()
    plt.savefig('./lstm/loss_%s_%s_%s_%s.png'%(window_size, hidden_units_lstm))
    plt.close('all')
    #Get the time it took to train the model (in seconds)
    training_time = int(math.floor(time.time() - start))
    return model, training_time



def test_model(model, X_test, Y_test, unnormalized_bases_tes, window_size, type_test, metal):
    """
    Test the model on the testing data
    
    Arguments:
    model -- The previously fitted 3 layer Recurrent Neural Network
    X_test -- A tensor of shape (267, 49, 35) that represents the x values of the testing data
    Y_test -- A tensor of shape (267,) that represents the y values of the testing data
    unnormalized_bases -- A tensor of shape (267,) that can be used to get unnormalized data points
    
    Returns:
    y_predict -- A tensor of shape (267,) that represnts the normalized values that the model predicts based on X_test
    real_y_test -- A tensor of shape (267,) that represents the actual spot prices throughout the testing period
    real_y_predict -- A tensor of shape (267,) that represents the model's predicted spot prices 
    fig -- A branch of the graph of the real predicted spot prices versus the real spot prices
    """
    #Test the model on X_Test
    y_predict_tes = model.predict(X_test)
    
    #Create empty 2D arrays to store unnormalized values
    real_y_test = np.zeros_like(Y_test)
    real_y_predict_tes = np.zeros_like(y_predict_tes)

    #Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    for i in range(0, Y_test.shape[0]):
        y_tes = Y_test[i]
        predict_tes = y_predict_tes[i]
        real_y_test[i] = (y_tes + 1)*unnormalized_bases_tes[i]
        real_y_predict_tes[i] = (predict_tes + 1)*unnormalized_bases_tes[i]
    return y_predict_tes, real_y_test, real_y_predict_tes
def price_change(Y_daybefore,  Y_test, y_predict,  window_size, type_test):
    """
    Calculate the percent change between each value and the day before
    
    Arguments:
    Y_daybefore -- A tensor of shape (267,) that represents the prices of each day before each price in Y_test
    Y_test -- A tensor of shape (267,) that represents the normalized y values of the testing data
    y_predict -- A tensor of shape (267,) that represents the normalized y values of the model's predictions
    
    Returns:
    Y_daybefore -- A tensor of shape (267, 1) that represents the prices of each day before each price in Y_test
    Y_test -- A tensor of shape (267, 1) that represents the normalized y values of the testing data
    delta_predict -- A tensor of shape (267, 1) that represents the difference between predicted and day before values
    delta_real -- A tensor of shape (267, 1) that represents the difference between real and day before values
    fig -- A plot representing percent change in spot price per day,
    """
    #Reshaping Y_daybefore and Y_test
    Y_daybefore = np.reshape(Y_daybefore, (-1, 1))
    Y_test = np.reshape(Y_test, (-1, 1))

    #The difference between each predicted value and the value from the day before
    delta_predict = (y_predict - Y_daybefore) / (1+Y_daybefore)

    #The difference between each true value and the value from the day before
    delta_real = (Y_test - Y_daybefore) / (1+Y_daybefore)

    #Plotting the predicted percent change versus the real percent change
    #fig = plt.figure(figsize=(10, 6))
    #ax = fig.add_subplot(111)
    #ax.set_title("Percent Change in Spot Price Per Day")
    #plt.plot(delta_predict, color='green', label = 'Predicted Percent Change')
    #plt.plot(delta_real, color='red', label = 'Real Percent Change')
    #plt.ylabel("Percent Change")
    #plt.xlabel("Time (Days)")
    #ax.legend()
    #plt.savefig('percentage_%s_%s_%s.png'%(str(type_test),str(hidden_units),str(window_size)))
    #plt.show()
    
    return Y_daybefore, Y_test, delta_predict, delta_real

def binary_price(delta_predict, delta_real):
    """
    Converts percent change to a binary 1 or 0, where 1 is an increase and 0 is a decrease/no change
    
    Arguments:
    delta_predict -- A tensor of shape (267, 1) that represents the predicted percent change in price
    delta_real -- A tensor of shape (267, 1) that represents the real percent change in price
    
    Returns:
    delta_predict_1_0 -- A tensor of shape (267, 1) that represents the binary version of delta_predict
    delta_real_1_0 -- A tensor of shape (267, 1) that represents the binary version of delta_real
    """
    #Empty arrays where a 1 represents an increase in price and a 0 represents a decrease in price
    delta_predict_1_0 = np.empty(delta_predict.shape)
    delta_real_1_0 = np.empty(delta_real.shape)

    #If the change in price is greater than zero, store it as a 1
    #If the change in price is less than zero, store it as a 0
    for i in range(delta_predict.shape[0]):
        if delta_predict[i][0] > 0:
            delta_predict_1_0[i][0] = 1
        else:
            delta_predict_1_0[i][0] = 0
    for i in range(delta_real.shape[0]):
        if delta_real[i][0] > 0:
            delta_real_1_0[i][0] = 1
        else:
            delta_real_1_0[i][0] = 0    

    return delta_predict_1_0, delta_real_1_0
def find_positives_negatives(delta_predict_1_0, delta_real_1_0):
    """
    Finding the number of false positives, false negatives, true positives, true negatives
    
    Arguments: 
    delta_predict_1_0 -- A tensor of shape (267, 1) that represents the binary version of delta_predict
    delta_real_1_0 -- A tensor of shape (267, 1) that represents the binary version of delta_real
    
    Returns:
    true_pos -- An integer that represents the number of true positives achieved by the model
    false_pos -- An integer that represents the number of false positives achieved by the model
    true_neg -- An integer that represents the number of true negatives achieved by the model
    false_neg -- An integer that represents the number of false negatives achieved by the model
    """
    #Finding the number of false positive/negatives and true positives/negatives
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(delta_real_1_0.shape[0]):
        real = delta_real_1_0[i][0]
        predicted = delta_predict_1_0[i][0]
        if real == 1:
            if predicted == 1:
                true_pos += 1
            else:
                false_neg += 1
        elif real == 0:
            if predicted == 0:
                true_neg += 1
            else:
                false_pos += 1
    return true_pos, false_pos, true_neg, false_neg
def calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict, Y_test):
    """
    Calculate various statistics to assess performance
    
    Arguments:
    true_pos -- An integer that represents the number of true positives achieved by the model
    false_pos -- An integer that represents the number of false positives achieved by the model
    true_neg -- An integer that represents the number of true negatives achieved by the model
    false_neg -- An integer that represents the number of false negatives achieved by the model
    Y_test -- A tensor of shape (267, 1) that represents the normalized y values of the testing data
    y_predict -- A tensor of shape (267, 1) that represents the normalized y values of the model's predictions
    
    Returns:
    precision -- How often the model gets a true positive compared to how often it returns a positive
    recall -- How often the model gets a true positive compared to how often is hould have gotten a positive
    F1 -- The weighted average of recall and precision
    Mean Squared Error -- The average of the squares of the differe
    nces between predicted and real values
    """
    #precision = float(true_pos) / (true_pos + false_pos)
    #recall = float(true_pos) / (true_pos + false_neg)
    acc = float(true_pos + true_neg)/(true_pos + false_pos + true_neg + false_neg)
    #F1 = float(2 * precision * recall) / (precision + recall)
    #Get Mean Squared Error
    MSE = mean_squared_error(y_predict.flatten(), Y_test.flatten())

    return MSE, acc
def rescale(y_norm, unnormalized_bases):
    real_price = np.zeros_like(y_norm)
    for i in range(0, y_norm.shape[0]):
        y_tes = y_norm[i]
        real_price[i] = (y_tes + 1)*unnormalized_bases[i]
    return real_price
def RMSE_test(Y_daybefore_val,Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, real_y_val, real_y_tes):
    Y_val_predict_last = rescale(Y_daybefore_val, unnormalized_bases_val)
    Y_tes_predict_last = rescale(Y_daybefore_tes, unnormalized_bases_tes)
    RMSE_val_last = np.sqrt(mean_squared_error(Y_val_predict_last.flatten(), real_y_val.flatten()))
    RMSE_tes_last = np.sqrt(mean_squared_error(Y_tes_predict_last.flatten(), real_y_tes.flatten()))
    STD_val = np.std(real_y_val)
    STD_tes = np.std(real_y_tes)

    return RMSE_val_last, RMSE_tes_last, STD_val, STD_tes

def conclusion(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size, metal_name,hidden_units_cnn, hidden_units_lstm, hidden_units_dense):
    y_predict_val, real_y_val, real_y_predict_val = test_model(model, X_val, Y_val, unnormalized_bases_val, window_size, 'val', metal_name)
    y_predict_tes, real_y_tes, real_y_predict_tes = test_model(model, X_test, Y_test, unnormalized_bases_tes, window_size, 'tes', metal_name)
    RMSE_val = np.sqrt(mean_squared_error(real_y_predict_val.flatten(), real_y_val.flatten()))
    RMSE_tes = np.sqrt(mean_squared_error(real_y_predict_tes.flatten(), real_y_tes.flatten()))
    val_0 = np.zeros_like(Y_val)
    test_0 = np.zeros_like(Y_test)
    MSE_0_val = mean_squared_error(val_0.flatten(), Y_val.flatten())
    MSE_0_tes = mean_squared_error(test_0.flatten(), Y_test.flatten())
    RMSE_val_last, RMSE_tes_last, STD_val, STD_tes = RMSE_test(Y_daybefore_val,Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, real_y_val, real_y_tes)
    Y_daybefore_val, Y_val, delta_predict_val, delta_real_val= price_change(Y_daybefore_val, Y_val, y_predict_val,  window_size, 'val')
    Y_daybefore_tes, Y_test, delta_predict_tes , delta_real_tes= price_change(Y_daybefore_tes, Y_test, y_predict_tes,  window_size, 'tes')
    delta_predict_1_0_val, delta_real_1_0_val = binary_price(delta_predict_val, delta_real_val)
    delta_predict_1_0_tes, delta_real_1_0_tes = binary_price(delta_predict_tes, delta_real_tes)
    f=open('./lstm/%s/result.txt'%(metal_name),'a')
    true_pos_val, false_pos_val, true_neg_val, false_neg_val = find_positives_negatives(delta_predict_1_0_val, delta_real_1_0_val)
    true_pos, false_pos, true_neg, false_neg = find_positives_negatives(delta_predict_1_0_tes, delta_real_1_0_tes)
    MSE_val, acc_val = calculate_statistics(true_pos_val, false_pos_val, true_neg_val, false_neg_val, y_predict_val, Y_val)
    MSE_tes, acc_tes = calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict_tes, Y_test)
    y_predict_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_predict_train.flatten(), Y_train.flatten())
    f.write(str(window_size)+' '+str(hidden_units_cnn)+' '+str(hidden_units_lstm)+' '+str(MSE_train)+' '+str(MSE_val)+' '+str(MSE_0_val)+' '+str(MSE_tes)+' '+str(MSE_0_tes)+' '+str(RMSE_val)+' '+str(RMSE_val_last)+' '+str(STD_val)+' '+str(RMSE_tes)+' '+str(RMSE_tes_last)+' '+str(STD_tes)+' '+str(acc_val)+' '+str(acc_tes)+'\n')

def train_model_conv_single(filename1, filename2, metal_name, hidden_units_cnn, hidden_units_lstm, hidden_units_dense):
    sequence_length = 11
    n_steps = 2
    #prepare the data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size = load_data(filename1, filename2, sequence_length)
    window = window_size/n_steps
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, window, 5))
    X_val = X_val.reshape((X_val.shape[0], n_steps, 1, window, 5))
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, window, 5))
    Y_train = Y_train.reshape((Y_train.shape[0], 1, 1))
    Y_val = Y_val.reshape((Y_val.shape[0], 1, 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1, 1))
    #build model
    model = convlstm_model(hidden_units_cnn, hidden_units_lstm, hidden_units_dense, window_size, 'linear', 'mse', 'adam')
    model.summary()
    model, training_time = fit_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, 512, 100, hidden_units_cnn, hidden_units_lstm, window_size, metal_name)
    conclusion(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size, metal_name, hidden_units_cnn, hidden_units_lstm, hidden_units_dense )
def train_model_lstm_single(filename1, filename2, metal_name, hidden_units_cnn, hidden_units_lstm, hidden_units_dense, sequence_length):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size = load_data(filename1, filename2, sequence_length)
    model = lstm_model(hidden_units_lstm, window_size, 'linear', 'mse', 'adam')
    model.summary()
    model, training_time = fit_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, 512, 100, hidden_units_cnn, hidden_units_lstm, hidden_units_dense, window_size, metal_name)
    conclusion(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes, window_size, metal_name, hidden_units_cnn, hidden_units_lstm, hidden_units_dense )

def read_all_metals(sequence_length):
    X_train1, Y_train1, X_val1, Y_val1, X_test1, Y_test1, Y_daybefore_val1, Y_daybefore_tes1, unnormalized_bases_val1, unnormalized_bases_tes1, window_size = load_data("LMAHDY.csv", "LMEAH3M.csv", sequence_length)
    X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2, Y_daybefore_val2, Y_daybefore_tes2, unnormalized_bases_val2, unnormalized_bases_tes2, window_size = load_data("LMCADY.csv", "LMECA3M.csv", sequence_length)     
    X_train3, Y_train3, X_val3, Y_val3, X_test3, Y_test3, Y_daybefore_val3, Y_daybefore_tes3, unnormalized_bases_val3, unnormalized_bases_tes3, window_size = load_data("LMNIDY.csv", "LMENI3M.csv",sequence_length)
    X_train4, Y_train4, X_val4, Y_val4, X_test4, Y_test4, Y_daybefore_val4, Y_daybefore_tes4, unnormalized_bases_val4, unnormalized_bases_tes4, window_size = load_data("LMPBDY.csv", "LMEPB3M.csv", sequence_length)
    X_train5, Y_train5, X_val5, Y_val5, X_test5, Y_test5, Y_daybefore_val5, Y_daybefore_tes5, unnormalized_bases_val5, unnormalized_bases_tes5, window_size = load_data("LMSNDY.csv", "LMESN3M.csv",sequence_length)
    X_train6, Y_train6, X_val6, Y_val6, X_test6, Y_test6, Y_daybefore_val6, Y_daybefore_tes6, unnormalized_bases_val6, unnormalized_bases_tes6, window_size = load_data("LMZSDY.csv", "LMEZS3M.csv",sequence_length)
    X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5,X_train6), axis = 0)
    Y_train = np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_train5,Y_train6), axis = 0)
    X_val = np.concatenate((X_val1,X_val2,X_val3,X_val4,X_val5,X_val6), axis = 0)
    Y_val = np.concatenate((Y_val1,Y_val2,Y_val3,Y_val4,Y_val5,Y_val6), axis = 0)
    X_test = np.concatenate((X_test1,X_test2,X_test3,X_test4,X_test5,X_test6), axis = 0)
    Y_test = np.concatenate((Y_test1,Y_test2,Y_test3,Y_test4,Y_test5,Y_test6), axis = 0)
    Y_daybefore_val = np.concatenate((Y_daybefore_val1, Y_daybefore_val2, Y_daybefore_val3, Y_daybefore_val4, Y_daybefore_val5, Y_daybefore_val6), axis = 0)
    Y_daybefore_tes = np.concatenate((Y_daybefore_tes1, Y_daybefore_tes2, Y_daybefore_tes3, Y_daybefore_tes4, Y_daybefore_tes5, Y_daybefore_tes6), axis = 0)
    unnormalized_bases_val = np.concatenate((unnormalized_bases_val1,unnormalized_bases_val2,unnormalized_bases_val3,unnormalized_bases_val4,unnormalized_bases_val5,unnormalized_bases_val6), axis = 0)
    unnormalized_bases_tes = np.concatenate((unnormalized_bases_tes1,unnormalized_bases_tes2,unnormalized_bases_tes3,unnormalized_bases_tes4,unnormalized_bases_tes5,unnormalized_bases_tes6), axis = 0)



    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_daybefore_val, Y_daybefore_tes, unnormalized_bases_val, unnormalized_bases_tes


            # #test model
# y_predict_val, real_y_val, real_y_predict_val = test_model(model, X_val, Y_val, unnormalized_bases_val, window_size, 'val', 'all')
# y_predict_tes, real_y_tes, real_y_predict_tes = test_model(model, X_test, Y_test, unnormalized_bases_tes, window_size, 'tes', 'all')
            
# y_predict_val1, real_y_val1, real_y_predict_val1 = test_model(model, X_val1, Y_val1, unnormalized_bases_val1, window_size, 'val', 'al')
# y_predict_val2, real_y_val2, real_y_predict_val2 = test_model(model, X_val2, Y_val2, unnormalized_bases_val2, window_size, 'val', 'cu')
# y_predict_val3, real_y_val3, real_y_predict_val3 = test_model(model, X_val3, Y_val3, unnormalized_bases_val3, window_size, 'val', 'ni')
# y_predict_val4, real_y_val4, real_y_predict_val4 = test_model(model, X_val4, Y_val4, unnormalized_bases_val4, window_size, 'val', 'pb')
# y_predict_val5, real_y_val5, real_y_predict_val5 = test_model(model, X_val5, Y_val5, unnormalized_bases_val5, window_size, 'val', 'sn')
# y_predict_val6, real_y_val6, real_y_predict_val6 = test_model(model, X_val6, Y_val6, unnormalized_bases_val6, window_size, 'val', 'zn')
            
# y_predict_tes1, real_y_tes1, real_y_predict_tes1 = test_model(model, X_test1, Y_test1, unnormalized_bases_tes1, window_size, 'tes', 'al')
# y_predict_tes2, real_y_tes2, real_y_predict_tes2 = test_model(model, X_test2, Y_test2, unnormalized_bases_tes2, window_size, 'tes', 'cu')
# y_predict_tes3, real_y_tes3, real_y_predict_tes3 = test_model(model, X_test3, Y_test3, unnormalized_bases_tes3, window_size, 'tes', 'ni')
# y_predict_tes4, real_y_tes4, real_y_predict_tes4 = test_model(model, X_test4, Y_test4, unnormalized_bases_tes4, window_size, 'tes', 'pb')
# y_predict_tes5, real_y_tes5, real_y_predict_tes5 = test_model(model, X_test5, Y_test5, unnormalized_bases_tes5, window_size, 'tes', 'sn')
# y_predict_tes6, real_y_tes6, real_y_predict_tes6 = test_model(model, X_test6, Y_test6, unnormalized_bases_tes6, window_size, 'tes', 'zn')

#             # #calculate RMSE
# RMSE_al_val = np.sqrt(mean_squared_error(real_y_predict_val1.flatten(), real_y_val1.flatten()))
# RMSE_cu_val = np.sqrt(mean_squared_error(real_y_predict_val2.flatten(), real_y_val2.flatten()))
# RMSE_ni_val = np.sqrt(mean_squared_error(real_y_predict_val3.flatten(), real_y_val3.flatten()))
# RMSE_pb_val = np.sqrt(mean_squared_error(real_y_predict_val4.flatten(), real_y_val4.flatten()))
# RMSE_sn_val = np.sqrt(mean_squared_error(real_y_predict_val5.flatten(), real_y_val5.flatten()))
# RMSE_zn_val = np.sqrt(mean_squared_error(real_y_predict_val6.flatten(), real_y_val6.flatten()))
# RMSE_al_tes = np.sqrt(mean_squared_error(real_y_predict_tes1.flatten(), real_y_tes1.flatten()))
# RMSE_cu_tes = np.sqrt(mean_squared_error(real_y_predict_tes2.flatten(), real_y_tes2.flatten()))
# RMSE_ni_tes = np.sqrt(mean_squared_error(real_y_predict_tes3.flatten(), real_y_tes3.flatten()))
# RMSE_pb_tes = np.sqrt(mean_squared_error(real_y_predict_tes4.flatten(), real_y_tes4.flatten()))
# RMSE_sn_tes = np.sqrt(mean_squared_error(real_y_predict_tes5.flatten(), real_y_tes5.flatten()))
# RMSE_zn_tes = np.sqrt(mean_squared_error(real_y_predict_tes6.flatten(), real_y_tes6.flatten()))

# val_0 = np.zeros_like(Y_val)
# test_0 = np.zeros_like(Y_test)
# MSE_0_val = mean_squared_error(val_0.flatten(), Y_val.flatten())
# MSE_0_tes = mean_squared_error(test_0.flatten(), Y_test.flatten())
         

# RMSE_al_val_last, RMSE_al_tes_last, STD_al_val, STD_al_tes = RMSE_test(Y_daybefore_val1,Y_daybefore_tes1, unnormalized_bases_val1, unnormalized_bases_tes1, real_y_val1, real_y_tes1)
# RMSE_cu_val_last, RMSE_cu_tes_last, STD_cu_val, STD_cu_tes = RMSE_test(Y_daybefore_val2,Y_daybefore_tes2, unnormalized_bases_val2, unnormalized_bases_tes2, real_y_val2, real_y_tes2)
# RMSE_ni_val_last, RMSE_ni_tes_last, STD_ni_val, STD_ni_tes = RMSE_test(Y_daybefore_val3,Y_daybefore_tes3, unnormalized_bases_val3, unnormalized_bases_tes3, real_y_val3, real_y_tes3)
# RMSE_pb_val_last, RMSE_pb_tes_last, STD_pb_val, STD_pb_tes = RMSE_test(Y_daybefore_val4,Y_daybefore_tes4, unnormalized_bases_val4, unnormalized_bases_tes4, real_y_val4, real_y_tes4)
# RMSE_sn_val_last, RMSE_sn_tes_last, STD_sn_val, STD_sn_tes = RMSE_test(Y_daybefore_val5,Y_daybefore_tes5, unnormalized_bases_val5, unnormalized_bases_tes5, real_y_val5, real_y_tes5)
# RMSE_zn_val_last, RMSE_zn_tes_last, STD_zn_val, STD_zn_tes = RMSE_test(Y_daybefore_val6,Y_daybefore_tes6, unnormalized_bases_val6, unnormalized_bases_tes6, real_y_val6, real_y_tes6)

# Y_daybefore_val, Y_val, delta_predict_val, delta_real_val= price_change(Y_daybefore_val, Y_val, y_predict_val,  window_size, 'val')
# Y_daybefore_tes, Y_test, delta_predict_tes , delta_real_tes= price_change(Y_daybefore_tes, Y_test, y_predict_tes,  window_size, 'tes')

# delta_predict_1_0_val, delta_real_1_0_val = binary_price(delta_predict_val, delta_real_val)
# delta_predict_1_0_tes, delta_real_1_0_tes = binary_price(delta_predict_tes, delta_real_tes)
# f=open('./lstm/result.txt','a')
# true_pos_val, false_pos_val, true_neg_val, false_neg_val = find_positives_negatives(delta_predict_1_0_val, delta_real_1_0_val)
# true_pos, false_pos, true_neg, false_neg = find_positives_negatives(delta_predict_1_0_tes, delta_real_1_0_tes)
# MSE_val, acc_val = calculate_statistics(true_pos_val, false_pos_val, true_neg_val, false_neg_val, y_predict_val, Y_val)
# MSE_tes, acc_tes = calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict_tes, Y_test)
# f.write(str(window_size)+' '+str(hidden_units_cnn)+' '+str(hidden_units_lstm)+' '+str(MSE_val)+' '+str(MSE_0_val)+' '+str(MSE_tes)+' '+str(MSE_0_tes)+' '+str(RMSE_al_val)+' '+str(RMSE_al_val_last)+' '+str(STD_al_val)+' '+str(RMSE_cu_val)+' '+str(RMSE_cu_val_last)+' '+str(STD_cu_val)+' '+str(RMSE_ni_val)+' '+str(RMSE_ni_val_last)+' '+str(STD_ni_val)+' '+str(RMSE_pb_val)+' '+str(RMSE_pb_val_last)+' '+str(STD_pb_val)+' '+str(RMSE_sn_val)+' '+str(RMSE_sn_val_last)+' '+str(STD_sn_val)+' '+str(RMSE_zn_val)+' '+str(RMSE_zn_val_last)+' '+str(STD_zn_val)+' '+str(RMSE_al_tes)+' '+str(RMSE_al_tes_last)+' '+str(STD_al_tes)+' '+str(RMSE_cu_tes)+' '+str(RMSE_cu_tes_last)+' '+str(STD_cu_tes)+' '+str(RMSE_ni_tes)+' '+str(RMSE_ni_tes_last)+' '+str(STD_ni_tes)+' '+str(RMSE_pb_tes)+' '+str(RMSE_pb_tes_last)+' '+str(STD_pb_tes)+' '+str(RMSE_sn_tes)+' '+str(RMSE_sn_tes_last)+' '+str(STD_sn_tes)+' '+str(RMSE_zn_tes)+' '+str(RMSE_zn_tes_last)+' '+str(STD_zn_tes)+' '+str(acc_val)+' '+str(acc_tes)+'\n')
