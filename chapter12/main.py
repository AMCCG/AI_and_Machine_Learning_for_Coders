import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import pathlib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import seaborn as sns
from keras.src.layers import Flatten, Conv2D, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array
from keras.src.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import urllib.request
import zipfile
from keras.src.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
import kagglehub


def convert_tflite_model():
    l0 = Dense(units=1)
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(l0)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
    ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
    model.fit(xs, ys, epochs=500)
    print(model.predict(np.array([10])))
    print('Here is what I learned: {}'.format(l0.get_weights()))
    export_dir = 'saved_model/1'
    # tf.saved_model.save(model, export_dir)
    model.export(export_dir)
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('model.tflite')
    tflite_model_file.write_bytes(tflite_model)


def load_tflite_model():
    export_dir = 'saved_model/1'
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    to_predict = np.array([[10]], dtype=np.float32)
    print(to_predict)
    interpreter.set_tensor(input_details[0]['index'], to_predict)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])
    print(tflite_result)
    pass


if __name__ == '__main__':
    # convert_tflite_model()
    load_tflite_model()
