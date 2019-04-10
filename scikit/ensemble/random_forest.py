import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('..')
import data_loader
from data_loader import *

classifier = RandomForestClassifier(
    n_estimators=20,
    min_samples_leaf=200
)
classifier.n_classes_ = 2

classifier.fit(train_data, label_data.ravel())

prediction = classifier.predict(test)

frame = pd.DataFrame(data=prediction)
frame.index += 1

frame.to_csv(
    "random_forest.csv", index=True, header=['Solution'], index_label='Id')

if __name__ == "__main__":
    pass