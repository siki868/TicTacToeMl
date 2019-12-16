import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import Counter


def load_data():
    x_moves, o_moves = np.load('x_moves.npy'), np.load('o_moves.npy')
    x_states, o_states = np.load('x_states.npy'), np.load('o_states.npy')

    y_train = []

    for move in x_moves:
        one_hot_action = np.zeros(9)
        one_hot_action[move] = 1
        y_train.append(one_hot_action)
    
    for move in o_moves:
        one_hot_action = np.zeros(9)
        one_hot_action[move] = 1
        y_train.append(one_hot_action)


    x_train = np.vstack((x_states, o_states))
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train



def generate_model():
    model = keras.models.Sequential([
        keras.layers.Dense(128, input_shape=(9, ), activation=tf.nn.relu),

        keras.layers.Dense(256, activation=tf.nn.relu),

        keras.layers.Dense(128, activation=tf.nn.relu),

        keras.layers.Dense(9, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    x_train, y_train = load_data()
    model = generate_model()
    model.fit(x_train, y_train, epochs=10)
    model.save('ttt1.h5')
