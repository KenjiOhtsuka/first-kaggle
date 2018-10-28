import pandas as pd
from sklearn.model_selection import train_test_split
from os import path

import numpy as np

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

DATA_DIR = path.dirname(__file__)
train = pd.read_csv(path.join(DATA_DIR, 'train.csv'), header=None)
train_label = pd.read_csv(path.join(DATA_DIR, 'trainLabels.csv'), header=None)
test = pd.read_csv(path.join(DATA_DIR, 'test.csv'), header=None)

train_x = train.values.astype('float32')  # numpy.ndarray
print(train_label.values.astype('int32'))
train_y = np.dot(train_label.values.astype('int32'), np.array([[-1, 1]])) + np.array([1, 0])  # numpy.ndarray

print(train_x.shape)
print(train_y.shape)
test_x = test.values.astype('float32')
print(test_x.shape)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y,
                                                    train_size=0.8)
print(x_train.shape)
print(y_train.shape)

n_in = len(x_train[0])
n_hidden = 200
n_out = len(y_train[0])

model = Sequential()

# model.add(Dense(n_hidden, input_dim=n_in))
# model.add(Activation('sigmoid'))
# model.add(Dense(n_out))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=0.01),
#               metrics=['accuracy'])

model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

# model.add(Dense(n_hidden))
# model.add(Activation('relu'))
#
# model.add(Dense(n_hidden))
# model.add(Activation('relu'))
#
model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

epochs = 8000
batch_size = 100

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

loss_and_metrics = model.evaluate(x_test, y_test)
print(loss_and_metrics)

## predict

print(test_x.shape)
prediction = model.predict(test_x)
prediction_labels = np.argmax(prediction, axis=1)
result = np.array([np.arange(len(prediction_labels)) + 1, prediction_labels],
                  dtype='int32').T
print(result)
np.savetxt('result.csv', result, delimiter=',', fmt='%d',
           header="Id,Solution", comments='')
