# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 01:29:46 2021

@author: rahiy
"""

from __future__ import absolute_import, division, print_function, unicode_literals 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 


print("shape of x_train", x_train.shape)
print("shape of y_train", y_train.shape)
print("shape of x_test", x_test.shape)
print("shape of y_test", y_test.shape)

x_val = x_train[:int(0.8*x_train.shape[0]),:]
x_train = x_train[int(0.8*x_train.shape[0]):,:]
y_val = y_train[:int(0.8*y_train.shape[0])]
y_train = y_train[int(0.8*y_train.shape[0]):]

print("shape of x_val", x_val.shape)
print("shape of x_train", x_train.shape)
print("shape of y_val", y_val.shape)
print("shape of y_train", y_train.shape)

model = tf.keras.models.Sequential([ 
 tf.keras.layers.Flatten(input_shape=(28, 28)), 
 tf.keras.layers.Dense(128, activation='relu'), 
 tf.keras.layers.Dropout(0.2), 
 tf.keras.layers.Dense(10) 
]) 

print(model.output.get_shape().as_list())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
loss=loss_fn, 
metrics=['accuracy']) 
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5) 

model.evaluate(x_test, y_test, verbose=2) 

model = tf.keras.models.Sequential([ 
 tf.keras.layers.Flatten(input_shape=(28, 28)), 
 tf.keras.layers.Dense(200, activation='relu'), 
 tf.keras.layers.Dropout(0.2), 
 tf.keras.layers.Dense(10) 
]) 

print(model.output.get_shape().as_list())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
loss=loss_fn, 
metrics=['accuracy']) 
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5) 

model.evaluate(x_test, y_test, verbose=2)
