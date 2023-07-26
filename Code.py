import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.utils import np_utils
import keras
from plot_keras_history import show_history, plot_history
from keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras import layers

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_1')
X = datadict["data"]
Y = datadict["labels"]

datadict_1 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_2')
X_1 = datadict_1["data"]
Y_1 = datadict_1["labels"]

datadict_2 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_3')
X_2 = datadict_2["data"]
Y_2 = datadict_2["labels"]

datadict_3 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_4')
X_3 = datadict_3["data"]
Y_3 = datadict_3["labels"]

datadict_4 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_5')
X_4 = datadict_4["data"]
Y_4 = datadict_4["labels"]

datadict_tb = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/test_batch')
X_tb = datadict_tb["data"]
Y_tb = datadict_tb["labels"]

x_merge= np.concatenate([X, X_1, X_2, X_3, X_4])
y_merge= np.concatenate([Y, Y_1, Y_2, Y_3, Y_4])


labeldict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = x_merge.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
Y = np.array(y_merge)

X_tb= X_tb.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8").astype('int')
Y_tb = np.array(Y_tb)

n_classes= 10
Y_1= np_utils.to_categorical(Y, n_classes) 
Y_tb1= np_utils.to_categorical(Y_tb, n_classes)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.99999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)

#nn model
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Dense(150, activation='relu'))
#model.add(Dense(50))
#model.add(layers.Dropout(0.5,noise_shape=None,seed=None))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model_history= model.fit(X, Y_1, validation_split=0.3, epochs=10, batch_size=60)
# model.summary() 
show_history(model_history)
plot_history(model_history, path="standard.png")
plt.close()

results = model.evaluate(X_tb, Y_tb1)
print(results)


# cnn model
cnn_model = Sequential()
cnn_model.add(keras.Input(shape=(32, 32, 3)))
cnn_model.add(layers.Conv2D(32, 5, strides = 1, data_format='channels_last',activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(keras.layers.MaxPool2D(pool_size=4, strides=2, padding='valid'))
# cnn_model.add(layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn_model.add(keras.layers.Dropout(0.5,noise_shape=None,seed=None))
# cnn_model.add(layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn_model.add(Conv2D(128, 5, activation='relu'))
cnn_model.add(keras.layers.Dropout(0.5,noise_shape=None,seed=None))
cnn_model.add(Conv2D(64, 5, activation='relu'))
cnn_model.add(Conv2D(32, 5, activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(20, activation='relu'))
cnn_model.add(Dense(10, activation='sigmoid'))

cnn_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model_history = cnn_model.fit(X, Y_1, validation_split=0.3, epochs=15, batch_size=100)
show_history(cnn_model_history)
# cnn_model.summary() 
plot_history(cnn_model_history, path="standard.png")
plt.close()

results_cnn = cnn_model.evaluate(X_tb, Y_tb1)
print(results_cnn)
