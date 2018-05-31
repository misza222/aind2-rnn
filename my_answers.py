import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    series_len = len(series)
    
    X = [series[start: start + window_size] for start in range(0, series_len - window_size)]
    y = [series[start + window_size]        for start in range(0, series_len - window_size)]
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    model = Sequential()
    
    model.add(Dense(32, input_shape=(window_size, 1)))
    
    lstm = LSTM(units=window_size, dropout=0.3, recurrent_dropout=0.1)
    model.add(lstm)
    
    output_layer = Dense(1)
    model.add(output_layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    
    def is_valid(i):
        return (i in punctuation) or (ord(i) >= 48 and ord(i) <= 57) or (ord(i) >= 97 and ord(i) <= 122)
    
    text = ''.join([i if is_valid(i) else '' for i in text])
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    text_len = len(text)
    
    inputs = [text[start:start+window_size] for start in range(0, text_len - window_size, step_size)]
    outputs= [text[start+window_size]       for start in range(0, text_len - window_size, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    lstm = LSTM(units=200, dropout=0.2, recurrent_dropout=0.1, input_shape=(window_size, num_chars))
    model.add(lstm)
    
    output_layer = Dense(num_chars, activation='softmax')
    model.add(output_layer)
    
    return model
