# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Bidirectional, Embedding
from sklearn.utils import class_weight
import tensorflow as tf
print(tf.__version__)
from sklearn.preprocessing import OneHotEncoder
import re
import random
import numpy as np
from os import path
import logging
from datetime import datetime
import json
from numpy.random import seed
import tensorflow as tf
from sklearn.metrics import classification_report

seed(1)
tf.random.set_seed(2)

uniq_fname = path.basename(__file__).split('.')[0] + datetime.now().strftime("_%Y%m%d-%H%M%S")
logging.basicConfig(filename=uniq_fname + '.log', format='%(levelname)s - %(message)s', level=logging.INFO, filemode='w')

def reporting(*args):
    print(" ".join(map(str,args)))
    logging.info(" ".join(map(str,args)))

NAME = "arabic_project"

program_start_time = datetime.now()
reporting(program_start_time.strftime("%Y-%m-%d %H:%M:%S"), 'Program start time')
reporting('Experiment: Classify poems, batch size 512')
reporting('The log file is', uniq_fname + '.log')
batch_size = 256 # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
reporting('Bach size', batch_size)
reporting('Epochs', epochs)
reporting('Latent dim', latent_dim)

data_path = weights_filename = None
data_path ="data.csv"

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

random.shuffle(lines)
input_texts = []
target_texts = []
input_characters = set()

for line in lines:
  try:
      if len(line) ==0 :
        continue
      target_text = str(list(line.split(","))[0])
      input_text = str(list(line.split(","))[1])
      input_text = str(input_text).replace("__len__", "").replace("    ", " ")
      input_texts.append(input_text.strip())
      target_texts.append(target_text)
      for char in input_text:
        if char not in input_characters:
                input_characters.add(char)
  except:
      print(line)

reporting('Number of samples:', len(input_texts))
max_seq_length = max([len(txt) for txt in input_texts])
reporting('Max sequence length:', max_seq_length)

input_characters = sorted(list(input_characters))
num_tokens = len(input_characters)
reporting('Number of tokens:', num_tokens)

input_token_index = dict( [(char, i) for i, char in enumerate(input_characters)])

input_data = np.zeros((len(input_texts), max_seq_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        input_data[i, t] = input_token_index[char] + 1.

encoder=OneHotEncoder(sparse=False)
out= np.array(target_texts).reshape(-1, 1)
output_data=encoder.fit_transform(out)
classes = output_data.shape[1]
reporting('Number of classes:', classes)
reporting("Class names:", list(encoder.categories_[0]))

model = Sequential()
model.add(Embedding(num_tokens+1, 32, input_length=max_seq_length, mask_zero=True))
model.add(Bidirectional(LSTM(latent_dim, input_shape=(None,num_tokens),
            dropout=0.1, recurrent_dropout=0.3),
            merge_mode='concat'))
model.add(Dense(output_data.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "training/cp.ckpt"
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy'),
]
if weights_filename:
    try:
        model.load_weights(weights_filename)
    except:
        reporting('*** Warning: Cannot read the weights file:', weights_filename)

test_samples = int(.15 * len(input_texts))

reporting('Train samples', len(input_texts)-test_samples)
reporting('Test samples', test_samples)

labels_train = np.argmax(output_data[test_samples:], axis=1)
class_weights = class_weight.compute_class_weight("balanced", np.unique(labels_train), labels_train)
class_weights = { i : class_weights[i] for i in range(0, len(class_weights) ) }
reporting("Class weights:", class_weights)

do_train = True
if do_train:
    training_start_time = datetime.now()
    reporting(training_start_time.strftime("%Y-%m-%d %H:%M:%S"), 'Training start time')
    history = model.fit(input_data[test_samples:], output_data[test_samples:], batch_size=batch_size, epochs=epochs,
                        validation_split=0.2, verbose=1, callbacks=callbacks_list, class_weight=class_weights,use_multiprocessing=True)#,use_multiprocessing=True, workers=8

    training_end_time = datetime.now()
    reporting(training_end_time.strftime("%Y-%m-%d %H:%M:%S"), 'Training end time')
    d = training_end_time - training_start_time
    reporting('Training time duration:', d.total_seconds(), 'seconds')

    reporting('Epoch', 'val_loss', 'val_acc', 'loss', 'acc')
    for i in range(len(history.history['accuracy'])):
        reporting('Epoch', i+1, ':', history.history['val_loss'][i],
                  history.history['val_accuracy'][i],
                  history.history['loss'][i],
                  history.history['accuracy'][i])

    model.load_weights(checkpoint_path)

model.save('model.h5')

prediction_start_time = datetime.now()
reporting(prediction_start_time.strftime("%Y-%m-%d %H:%M:%S"), 'Testing ...')
scores = model.evaluate(input_data[0:test_samples], output_data[0:test_samples])
reporting('Loss on the test set', scores[0])
reporting('Accuracy on the test set', scores[1])

y_pred = model.predict(input_data[0:test_samples])
decoded_y_test = output_data[0:test_samples].argmax(axis=1)
decoded_y_pred = y_pred.argmax(axis=1)

class_names = ''
with open("report.log", 'w', encoding='utf-8') as w :
 for i in range(classes):
        class_names = str(i)+':' + encoder.categories_[0][i] + ''
        w.write( class_names+"\n")

 print(classification_report(decoded_y_test, decoded_y_pred, digits=4))

program_end_time = datetime.now()
reporting(program_end_time.strftime("%Y-%m-%d %H:%M:%S"), 'Program end time')
d = program_end_time - program_start_time
reporting('Program time duration:', d.total_seconds(), 'seconds')