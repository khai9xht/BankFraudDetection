import keras
from keras import Sequential
from keras.layers import Dense
import keras.backend as K


def build_model(input_shape, out_shape):
    model = Sequential([
        Dense(input_shape, input_shape=(input_shape, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(out_shape, activation='softmax')
    ])
    return model