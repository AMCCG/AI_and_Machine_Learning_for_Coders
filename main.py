import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns
from keras.src.layers import Flatten
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(np.__version__)
    print(pd.__version__)
    print(tf.__version__)
    print(keras.__version__)
    print(sns.__version__)
    print(StandardScaler())
    print(RandomOverSampler())


def Lession1():
    l0 = Dense(units=1)
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(l0)
    model.compile(optimizer="sgd", loss="mean_squared_error")
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    model.fit(xs, ys, epochs=500)
    print(model.predict(np.array([10])))
    print("Here is what i learned: {}".format(l0.get_weights()))


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print('\nReached 95% accuracy so cancelling training!')
            self.model.stop_training = True


def Lession2():
    callbacks = myCallback()
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    print(len(training_images[0]))
    print(len(training_images[0][0]))
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == '__main__':
    print_hi('PyCharm')
    Lession1()
    Lession2()
