import pandas as pd
from sklearn import linear_model
import sys
sys.path.append('..')
import data_loader
from data_loader import *


reg = linear_model.LogisticRegression()
reg.fit(train_data, label_data.ravel())

prediction = reg.predict(test)
prediction = (prediction > 0.5) * 1

frame = pd.DataFrame(data=prediction)
frame.index += 1

frame.to_csv(
    "logistic_classifier.csv", index=True, header=['Solution'], index_label='Id')

if __name__ == "__main__":
    pass