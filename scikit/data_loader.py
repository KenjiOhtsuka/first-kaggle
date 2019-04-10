import pandas as pd
import numpy as np

train = pd.read_csv(
    "../train.csv", header=None, delimiter=",", quoting=3
)
train_data = np.array(train)

label = pd.read_csv(
    "../trainLabels.csv", header=None, delimiter=",", quoting=3
)
label_data = np.array(label)

test = pd.read_csv(
    "../test.csv", header=None, delimiter=",", quoting=3
)
test_data = np.array(test)
