import pandas as pd
import os
import argparse
import datetime
import numpy as np
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras_diagram import ascii

class basic_linear_regression_model():
    def __init__(self,npz_file,n_layers,n_node,n_batch,n_epoch):
        self.npz_file = npz_file
        self.n_layers = n_layers
        self.n_node = n_node
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.trained_network = []
        self.loss_on_held_out = 0
        self.data_struct = np.load(self.npz_file)


    def basic_model(self):
        model = Sequential()
        # input layer
        model.add(Dense(self.n_node[0],input_dim = 300, kernel_initalizer = 'normal',activation = 'relu'))

        # hidden layers
        for i in range(1,self.n_layers):
            model.add(Dense(self.n_node[i], kernel_initializer='normal', activation='relu'))

        # output layer
        model.add(Dense(2), kernel_initalizer='normal')

        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    def fit_model(self):
        model_fit = []
        model_fit.append(
            ('mlp', KerasRegressor(build_fn=self.basic_model, epochs=self.n_epoch, batch_size=self.n_batch, verbose=1)))
        self.trained_network = Pipeline(model_fit)
        self.trained_network.fit(self.X,self.target)

    def print_model(self):
        print(ascii(self.model))

    def evaluate(self):
        y_hat = self.trained_network.predict(self.held_out_data)
        self.eval = np.mean(np.abs(self.held_out_y - y_hat))


