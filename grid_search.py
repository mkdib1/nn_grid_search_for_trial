# -*- coding: utf-8 -*-

# LIBRARIES & PACKAGES


import itertools
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, BatchNormalization, Reshape, Input, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Setting seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# FUNCTIONS

def gaussian_noise(signal, sigma):
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise

def time_reverse(signal, p_aug_rev):
    if np.random.rand() < p_aug_rev:
        return np.flip(signal) # reverse the order of elements in an array passed as param
    else:
        return signal

def sign_flip(signal, p_aug_flip):
    if np.random.rand() < p_aug_flip:
        return -signal  # flipping the sign
    else:
        return signal

def data_augmented(signal, sigma, p_aug_rev, p_aug_flip):
    signal = gaussian_noise(signal, sigma)
    signal = time_reverse(signal, p_aug_rev)
    signal = sign_flip(signal, p_aug_flip)
    return signal

def compile_model(model):

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])


def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False)
    callbacks = [early_stopping]

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))

    return history

def evaluate_model(model, X_test, y_test):

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')
    return loss, accuracy

# DATASET LOADING & VARIABLES SETTING

# loading dataset
eeg = pd.read_csv('./data/eeg.csv', sep=";")

X = eeg.drop(columns=['class'])
y = eeg['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_plt = y_train.copy()
y_test_plt = y_test.copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def create_model():

  inputs = Input(shape=(X_train.shape[1], 1))

  x = Conv1D(filters=1024, kernel_size=4, strides=2, padding='same')(inputs)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=2)(x)
  x = Dropout(0.3)(x)

  x = Conv1D(filters=512, kernel_size=4, strides=2, padding='same')(inputs)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=2)(x)
  x = Dropout(0.3)(x)

  x = Conv1D(filters=256, kernel_size=4, strides=2, padding='same')(inputs)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=2)(x)
  x = Dropout(0.3)(x)

  x = Conv1D(filters=128, kernel_size=4, strides=2, padding='same')(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=2)(x)
  x = Dropout(0.3)(x)

  x = Flatten()(x)

  x = Dense(128)(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = Reshape((1, 128))(x)

  x = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)

  x = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)

  x = LSTM(32, return_sequences=False, recurrent_dropout=0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)


  outputs = Dense(5, activation='softmax')(x)

  model = Model(inputs=inputs, outputs=outputs)


  return model

# defining 3*3 matrix of values which will be used for grid search tuning
sigma_values = [0.01, 0.05, 0.1]
p_aug_rev_values = [0, 0.2, 0.5, 0.8]
p_aug_flip_values = [0, 0.2, 0.5, 0.8]

param_grid = list(itertools.product(sigma_values, p_aug_rev_values, p_aug_flip_values))

best_accuracy = 0
best_params = None
results = []

# count = 27
for sigma, p_aug_rev, p_aug_flip in param_grid:
    X_train_augm = np.apply_along_axis(lambda row: data_augmented(row, sigma, p_aug_rev, p_aug_flip), axis=1, arr=X_train)
    model_hna = create_model()
    compile_model(model_hna)
    history_hna = train_model(model_hna, X_train_augm, y_train, X_test, y_test)
    _, acc = evaluate_model(model_hna, X_test, y_test)
    results.append((sigma, p_aug_rev, p_aug_flip, acc))
    # count -= 1
    # print(f"Mancano {count} addestramenti")

    if acc  > best_accuracy:
        best_accuracy = acc
        best_params = (sigma, p_aug_rev, p_aug_flip)

print("Best parameters:")
print("Sigma:", best_params[0])
print("Time Reverse Probability:", best_params[1])
print("Sign Flip Probability:", best_params[2])
print("Best Accuracy:", best_accuracy)