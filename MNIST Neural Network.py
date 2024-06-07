#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:09:40 2024

@author: christian_nguyen
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# downloads the MNIST Data Set
from tensorflow.keras.datasets import mnist
#allows graphing functionalities
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#-1 means keep whatever values, and 784 is 28*28
#.astype("float32) reduces computation req
#/255 normalizes the data
x_train= x_train.reshape(-1,28*28).astype("float32")/255.0
x_test= x_test.reshape(-1,28*28).astype("float32")/255.0

# Sequential API (convenient but inflexible)
#model = keras.Sequential(
    # 512 is the number of nodes in the first layer of network
    # relu activation function
    # last layer doesn't need an activation function
    #[keras.Input(shape=(28*28)),
     #layers.Dense(1028, activation='relu'),
     #layers.Dense(526, activation='relu'),
     #layers.Dense(10),   
     #]  
    #)

#You can also write a sequential model one layer at a time
model = keras.Sequential()
model.add(keras.Input(shape=(784))) 
model.add(layers.Dense(1208,activation='relu'))
model.add(layers.Dense(526,activation='relu'))
model.add(layers.Dense(10))
#layer extraction
model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output])
feature = model.predict(x_train)
print(feature.shape)
print(model.summary())
# Gives Data regarding the model
model.compile(
    #labels are just integer for correct label
    # goes through losses first, then sparse categorical 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=["accuracy"],
    )
#verbose = 2 prints every time
history=model.fit(x_train,y_train,batch_size=32,epochs=10, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=32,verbose=2)


plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

