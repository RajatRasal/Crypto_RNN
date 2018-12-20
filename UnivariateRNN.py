import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, CuDNNLSTM
from tensorflow.keras.models import load_model
from tensorflow.keras import backend


def build_model(neurons, hidden_neurons, keep_prob, layers, data_dim,
                output_size, cell, optimizer, loss):
    print("Cell:", cell)
    model = Sequential()
    model.add(cell(neurons, return_sequences=True, input_shape=(None, data_dim)))
    model.add(Dropout(keep_prob))
    for _ in range(layers-1):
        model.add(cell(hidden_neurons, return_sequences=True, input_shape=(None, data_dim)))
        model.add(Dropout(keep_prob))
    model.add(cell(neurons))
    model.add(Dropout(keep_prob))
    model.add(Dense(units=output_size))
    model.compile(optimizer=adam, loss=loss)
    return model


class UnivariateRNN(BaseEstimator, RegressorMixin):
    def __init__(self, neurons=5, hidden_neurons=2, keep_prob=0.5, layers=3,
                 epochs=5, batch_size=100, cell=LSTM, output_size=1, 
                 data_dim=1, optimizer='adam', loss='mse'):
        self.model = None
        self.cell = cell
        
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.keep_prob = keep_prob
        self.layers = layers
        self.output_size = output_size
        self.data_dim = data_dim
        
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        
    def get_params(self, deep=True):
        params = vars(self)
        del params['output_size']
        del params['data_dim']
        del params['model']
        return params
    
    def set_params(self, **params):
        self.model = None
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        assert X is not None and y is not None
        if not self.model:
            self.model = build_model(self.neurons, self.hidden_neurons, 
                                     self.layers, self.keep_prob,
                                     self.data_dim, self.output_size, self.cell)
        hist = self.model.fit(X_train, y_train, batch_size=batch_size,
                              epochs=epochs, validation_split=0.3, verbose=1)
        return hist
    
    def predict(self, X):
        assert X is not None
        if not self.model:
            self.model = build_model(self.neurons, self.hidden_neurons,
                                     self.layers, self.keep_prob,
                                     self.data_dim, self.output_size, self.cell)
        return self.model.predict(X).flatten()
