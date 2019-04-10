import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


def print_title(title: str):
    print(
        "################################################################################")
    print(f"# {title}")
    print(
        "################################################################################")


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


################################################################################

print_title("Shape")
print(train.shape)

print_title("values")
print(train.columns.values)

print_title("download")
# nltk.download()

print_title("num reviews")
num_reviews = train["review"].size

################################################################################

print_title("Cleaning and parsing the training set movie reviews ...")
clean_train_reviews = []
for i in range(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print(stopwords.words("english"))

print(review_to_words(train["review"][0]))

################################################################################

print_title("Create bag of words")
vectorizer = CountVectorizer(
    analyzer="word", tokenizer=None, preprocessor=None,
    stop_words=None, max_features=5000
)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print(vocab)

dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)

################################################################################

print_title("Train Random Forest")

# Initialize random forest classifier with 100 trees.
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

################################################################################

test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
print(test.shape)

num_reviews = len(test["review"])
clean_test_reviews = []

print_title("Clean and parse the last set movie reviews")
for i in range(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d" % (i + 1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

if __name__ == "__main__":
    pass
