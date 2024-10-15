import os
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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


def Lession3(name):
    print(name)
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    print(len(training_images[0]))
    print(len(training_images[0][0]))
    print(len(training_images))
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])
    model.summary()


def Lession3_ZipFile():
    file_name = 'horse-or-human.zip'
    validation_file_name = 'horse-or-human.zip'
    training_dir = 'horse-or-human/training/'
    validation_dir = 'horse-or-human/validation/'
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()
    zip_ref = zipfile.ZipFile(validation_file_name, 'r')
    zip_ref.extractall(validation_dir)
    zip_ref.close()
    pass


def Lession3_ImageDataGenerator():
    training_dir = 'horse-or-human/training/'
    train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    validation_dir = 'horse-or-human/validation/'
    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # 304 = (16 * 10) + (16 * 9) + (16 * 9)
    # 4640 =  32 + (32 * (16 * 9))
    # 18,496 = 64 + (64 * (32*9))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
    path = 'horse-or-human/testing/horse-1.jpg'
    img = load_img(path, target_size=(300, 300))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print("{0:.2f}".format(np.array(classes[0])[0]))
    if (classes[0] > 0.5):
        print("It is s human")
    else:
        print("It is s horse")
    print("*************************")


def lession3_transfer_learning():
    print("transfer_learning")
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(weights_file)
    pre_trained_model.summary()
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed7')
    # print('last layer output shape: ',last_layer.output_shape)
    last_output = last_layer.output
    x = Flatten()(last_output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(pre_trained_model.input, x)
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    training_dir = 'horse-or-human/training/'
    train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        batch_size=20,
        target_size=(150, 150),
        class_mode='binary'
    )
    validation_dir = 'horse-or-human/validation/'
    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        target_size=(150, 150),
        class_mode='binary'
    )
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        verbose=1)


def lession3_multiclass_classification():
    print('multiclass')

    # Download latest version
    # path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    # print("Path to dataset files:", path)
    local_zip = 'tmp/rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()
    local_zip = 'tmp/rps-test-set.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()
    rock_dir = os.path.join('/tmp/rps/rock')
    paper_dir = os.path.join('/tmp/rps/paper')
    scissors_dir = os.path.join('/tmp/rps/scissors')
    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))
    TRAINING_DIR = "/tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "/tmp/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode='categorical'
    )
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)
    model.save("rps.h5")


def lession3_dogs_cats():
    print(len(os.listdir('tmp/PetImages/Cat/')))
    print(len(os.listdir('tmp/PetImages/Dog/')))
    try:
        os.mkdir('tmp/cats-v-dogs')
        os.mkdir('tmp/cats-v-dogs/training')
        os.mkdir('tmp/cats-v-dogs/testing')
        os.mkdir('tmp/cats-v-dogs/training/cats')
        os.mkdir('tmp/cats-v-dogs/training/dogs')
        os.mkdir('tmp/cats-v-dogs/testing/cats')
        os.mkdir('tmp/cats-v-dogs/testing/dogs')
    except OSError:
        print(OSError.errno)
        pass
    split_size = .9
    CAT_SOURCE_DIR = "tmp/PetImages/Cat/"
    TRAINING_CATS_DIR = "tmp/cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "tmp/cats-v-dogs/testing/cats/"
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    DOG_SOURCE_DIR = "tmp/PetImages/Dog/"
    TRAINING_DOGS_DIR = "tmp/cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "tmp/cats-v-dogs/testing/dogs/"
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
    print(len(os.listdir('tmp/cats-v-dogs/training/cats/')))
    print(len(os.listdir('tmp/cats-v-dogs/training/dogs/')))
    print(len(os.listdir('tmp/cats-v-dogs/testing/cats/')))
    print(len(os.listdir('tmp/cats-v-dogs/testing/dogs/')))
    pass

def split_data(source, training, testing, split_size):
    files = []
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]
    for filename in training_set:
        this_file = source + filename
        destination = training + filename
        copyfile(this_file, destination)
    for filename in testing_set:
        this_file = source + filename
        destination = testing + filename
        copyfile(this_file, destination)
    pass


if __name__ == '__main__':
    # print_hi('PyCharm')
    # Lession1()
    # Lession2()
    # Lession3('CNN')
    # Lession3_ZipFile()
    # Lession3_ImageDataGenerator()
    # lession3_transfer_learning()
    lession3_dogs_cats()
    # lession3_multiclass_classification()
