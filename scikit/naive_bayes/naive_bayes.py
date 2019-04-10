import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sys
sys.path.append('..')
import data_loader
from data_loader import *


gnb = GaussianNB()
gnb.fit(train_data, label_data.ravel())

prediction = gnb.predict(test)

frame = pd.DataFrame(data=prediction)
frame.index += 1

frame.to_csv(
    "naive_bayes.csv", index=True, header=['Solution'], index_label='Id')

if __name__ == "__main__":
    pass