import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import Counter


def load_data():
    """
    y_train - one hot arrays out of 9 possible moves
    x_moves and o_moves are stacked because the model learns from both x and o valid moves
    """
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
    """
    Simple NN, mozda treba promenim jer je shit
    """
    model = keras.models.Sequential([
        keras.layers.Dense(128, input_shape=(9, ), activation=tf.nn.relu),
        
        keras.layers.Dropout(0.2),

        keras.layers.Dense(256, activation=tf.nn.relu),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(128, activation=tf.nn.relu),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(9, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def train_new_model():
    x_train, y_train = load_data()
    model = generate_model()
    model.fit(x_train, y_train, epochs=5, batch_size=50)
    model.save('ttt6k.h5')

if __name__ == "__main__":
    train_new_model()
