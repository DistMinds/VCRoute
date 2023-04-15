import numpy as np

# Random sampling
import random

# Keras API
from tensorflow import keras

# Deep learning
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
from keras import losses
from keras.callbacks import ModelCheckpoint

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of
    deep learning models
    """
    # Extracting the idx of features that are passed from the array
    n_features = ts.shape[1]

    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an RNN input shape
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y


class NNMultistepModel:

    def __init__(
            self,
            X,
            Y,
            n_outputs,
            n_lag,
            n_ft,
            n_layer,
            batch,
            epochs,
            Xval=None,
            Yval=None,

            file_path='best_checkpoint.hdf5'
    ):
        # 搭建LSTM模型，预测
        self.model = Sequential()
        # LSTM 第一层
        self.model.add(LSTM(n_layer, return_sequences=True, input_shape=(n_lag, n_ft)))
        self.model.add(Dropout(0.2))

        # LSTM 第二层
        self.model.add(LSTM(n_layer, return_sequences=True))
        self.model.add(Dropout(0.2))

        # LSTM 第三层
        self.model.add(LSTM(n_layer))
        self.model.add(Dropout(0.2))

        # Dense层
        self.model.add(Dense(units=n_outputs))

        self.batch = batch
        self.epochs = epochs
        self.n_layer = n_layer
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.file_path = file_path

    def trainCallback(self):
        return ModelCheckpoint(filepath=self.file_path,
                               monitor='loss',
                               mode='min',
                               save_best_only=True,
                               save_weights_only=True)

    def valCallback(self):
        return ModelCheckpoint(filepath=self.file_path,
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True,
                               save_weights_only=True)

    def train(self):
        # Getting the untrained model
        empty_model = self.model

        # Compiling the model
        empty_model.compile(loss='mae', optimizer='adam')

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.Xval, self.Yval),
                shuffle=False,
                callbacks=[self.valCallback()]
            )
        else:
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=False,
                callbacks=[self.trainCallback()]
            )

        # Saving to original model attribute in the class
        self.model = empty_model

        # Returning the training history
        return history

    def predict(self, X):
        return self.model.predict(X)


