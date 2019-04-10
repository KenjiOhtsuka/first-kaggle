import pandas as pd
from sklearn import tree
import sys
sys.path.append('..')
import data_loader
from data_loader import *


reg = tree.DecisionTreeClassifier()
reg.fit(train_data, label_data.ravel())

prediction = reg.predict(test)

frame = pd.DataFrame(data=prediction)
frame.index += 1

frame.to_csv(
    "decision_tree.csv", index=True, header=['Solution'], index_label='Id')

if __name__ == "__main__":
    pass