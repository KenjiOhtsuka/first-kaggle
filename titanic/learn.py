import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re

test_file_path = 'test.csv'
learn_file_path = 'train.csv'

df = pd.read_csv(learn_file_path)

def name_title(name):
    mr = re.compile("Mr.")
    miss = re.compile("Miss.")
    mrs = re.compile("Mrs.")
    if mr.search(name) != None:
        return 1
    elif miss.search(name) != None:
        return 2
    elif mrs.search(name) != None:
        return 3
    else:
        return 0

def convert(data_x):
    data_x['Fare'] = data_x['Fare'].fillna(data_x['Fare'].median())
    data_x['Age'] = data_x['Age'].fillna(data_x['Age'].median())
    data_x['Embarked'] = data_x['Embarked'].fillna('S')
    data_x['Sex'] = data_x['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    data_x['Embarked'] = data_x['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    data_x['Name'] = data_x['Name'].apply(lambda x: name_title(x))
    data_x = data_x.drop(['Cabin','PassengerId','Ticket'],axis=1)
    return data_x

df = convert(df)
train_X = df.drop('Survived', axis=1)
train_y = df.Survived
(train_X, test_X, train_y, test_y) = train_test_split(train_X, train_y, test_size=0.2, random_state=666)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)

#for i in range(3, 10):
#    for j in range(3, 10):

i = 6
j = 8
clf = RandomForestClassifier(random_state=0, max_depth=i, max_features=3)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
print(i, j)
print(accuracy_score(pred, test_y))

est_X = pd.read_csv('test.csv')
pid_column = est_X.PassengerId
est_X = convert(est_X)
ans_y = clf.predict(est_X)

ans_y = np.hstack((np.array(pid_column).reshape(-1, 1), ans_y.reshape(-1, 1)))
np.savetxt('ans.csv', ans_y, fmt="%d,%d", delimiter=',', header="PassengerId,Survived", comments='')

if 1 != 2:
    sys.exit()
from sklearn import svm

def name_title(name):
    mr = re.compile("Mr.")
    miss = re.compile("Miss.")
    mrs = re.compile("Mrs.")
    if mr.search(name) != None:
        return 1
    elif miss.search(name) != None:
        return 2
    elif mrs.search(name) != None:
        return 3
    else:
        return 0

df = pd.read_csv("train.csv")
df = convert(df)
train_X = df.drop('Survived', axis=1)
train_y = df.Survived
(train_X, test_X, train_y, test_y) = train_test_split(train_X, train_y, test_size=0.2, random_state=666)

print(train_X.head(5))
clf = svm.SVC(gamma=0.1)
clf.fit(train_X, train_y)
est_X = pd.read_csv('test.csv')
pid_column = est_X.PassengerId
est_X = convert(est_X)
ans_y = clf.predict(est_X)
ans_y = np.hstack((np.array(pid_column).reshape(-1, 1), ans_y.reshape(-1, 1)))
np.savetxt('ans_svm.csv', ans_y, fmt="%d,%d", delimiter=',', header="PassengerId,Survived", comments='')

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

df = pd.read_csv("train.csv")
df = convert(df)
train_X = df.drop('Survived', axis=1).values.astype('float32')
train_y = np.identity(2)[df.Survived.values.astype('int32')]
(train_X, test_X, train_y, test_y) = train_test_split(train_X, train_y, test_size=0.2, random_state=666)

n_in = len(train_X[0])
n_hidden = 200
n_out = len(test_y[0])

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))
model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))
model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))
model.add(Dense(n_out))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
epochs = 1000
batch_size = 100

print(train_X)
print(train_y)

model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)

loss_and_metrics = model.evaluate(test_X, test_y)
print(loss_and_metrics)

## predict

test_X = pd.read_csv('test.csv')
pid_column = test_X.PassengerId
test_X = convert_for_svm(test_X)
print(test_X.shape)
prediction = model.predict(test_X)
prediction_labels = np.argmax(prediction, axis=1)
result = np.array([np.arange(len(prediction_labels)) + 1, prediction_labels],
                  dtype='int32').T
print(result[:, 1].shape)
print(np.array(pid_column).reshape(-1, 1).shape)
r = np.hstack((np.array(pid_column).reshape(-1, 1), result[:, 1].reshape(-1, 1)))
np.savetxt('result.csv', r, delimiter=',', fmt='%d',
           header="PassengerId,Survived", comments='')
