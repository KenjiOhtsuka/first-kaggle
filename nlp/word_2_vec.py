import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data

train = pd.read_csv(
    "labeledTrainData.tsv", header=0, delimiter='\t', quoting=3
)

test = pd.read_csv(
    "testData.tsv", header=0, delimiter='\t', quoting=3
)

unlabeled_train = pd.read_csv(
    "unlabeledTrainData.tsv", header=0, delimiter='\t', quoting=3
)

print("Read %d labeled train reviews.", train['review'].size)
print("Read %d labeled test reviews.", test['review'].size)
print("Read %d unlabeled train reviews.", unlabeled_train['review'].size)


################################################################################

def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


################################################################################

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


sentences = []
for review in train['review']:
    sentences += review_to_sentences(review, tokenizer)
for review in unlabeled_train['review']:
    sentences += review_to_sentences(review, tokenizer)

################################################################################

# Architecture:
#     Architecture options are skip-gram (default) or continuous bag of words.
#     We found that skip-gram was very slightly slower but produced better results.
# Training algorithm:
#     Hierarchical softmax (default) or negative sampling.
#     For us, the default worked well.
# Downsampling of frequent words:
#     The Google documentation recommends values between .00001 and .001.
#     For us, values closer 0.001 seemed to improve
#     the accuracy of the final model.
# Word vector dimensionality:
#     More features result in longer runtimes, and often, but not always,
#     result in better models.
#     Reasonable values can be in the tens to hundreds; we used 300.
# Context / window size:
#     How many words of context should the training algorithm take into account?
#     10 seems to work well for hierarchical softmax (more is better, up to a point).
# Worker threads:
#     Number of parallel processes to run. This is computer-specific,
#     but between 4 and 6 should work on most systems.
# Minimum word count:
#     This helps limit the size of the vocabulary to meaningful words.
#     Any word that does not occur at least this many times across all documents
#     is ignored. Reasonable values could be between 10 and 100.
#     In this case, since each movie occurs 30 times,
#     we set the minimum word count to 40, to avoid attaching
#     too much importance to individual movie titles. This resulted in
#     an overall vocabulary size of around 15,000 words.
#     Higher values also help limit run time.

import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300  # Word vector dimentionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

from gensim.models import word2vec

model = word2vec.Word2Vec(
    sentences, workers=num_workers, size=num_features, min_count=min_word_count,
    window=context, sample=downsampling
)

model_name = "300features_40minwords_10context"
model.save(model_name)

print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.doesnt_match("paris berlin london austria".split()))

print(model.most_similar("man"))
print(model.most_similar("queen"))
print(model.most_similar("awful"))

################################################################################
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

print(type(model.wv.syn0))
print(type(model.wv.syn0.shape))

print(model['flower'])

################################################################################

import numpy as np


def makeFeatureVec(words, model, num_features):
    """
    Function to average all of the word vectors in a given paragraph

    :param words:
    :param model:
    :param num_features:
    :return:
    """

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set. for speed.
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vacabulary, add its feature vector to the total.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    """
    Given a set of reviews (each one a list of words), calculate
    the average feature vector for each one and return a 2D array

    :param reviews:
    :param model:
    :param num_features:
    :return:
    """
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    # Loop through the reviews
    for review in reviews:
        if counter % 1000. == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        counter = counter + 1

    return reviewFeatureVecs


################################################################################

# Calculate average feature vectors for training and testing sets,
# using the fucntions we defined above. Notice that we now use stop word
# removal.


clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(
        review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(
        review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

################################################################################

# Fit a random forest to the training data

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(trainDataVecs, train['sentiment'])

result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id": test['id'], 'sentiment': result})
output.to_csv('Word2Vec_AverageVectors.csv', index=False, quoting=3)

################################################################################

from sklearn.cluster import KMeans
import time

start = time.time()
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)

kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()
elapsed = end - start
print("Time taken for KMeans clustering", elapsed, "seconds.")

word_centroid_map = dict(zip(model.wv.index2word, idx))

for cluster in range(0, 10):
    print("Clustor %d" % cluster)
    words = []
    for i in range(0, len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] == cluster:
            words.append(list(word_centroid_map.keys())[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1

    bag_of_centroids = np.zeros(num_centroids, dtype='float32')
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


################################################################################

train_centroids = np.zeros(
    (train['review'].size, num_clusters), dtype='float32'
)

counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

test_centroids = np.zeros(
    (train["review"].size, num_clusters), dtype='float32'
)

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1


forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data ...")
forest = forest.fit(train_centroids, train['sentiment'])
forest = forest.predict(test_centroids)

output = pd.DataFrame(data={"id":test['id'], "sentiment":result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)


if __name__ == '__main__':
    pass
