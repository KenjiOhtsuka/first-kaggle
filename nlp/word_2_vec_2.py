from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10contezt")

print(type(model.syn0))
print(type(model.syn0.shape))

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
    index2word_set = set(model.index2word)

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
        review_to_sentences(review, remove_stopwords=True))

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
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()
elapsed = end - start
print("Time taken for KMeans clustering", elapsed, "seconds.")

word_centroid_map = dict(zip(model.index2word, idx))

for cluster in range(0, 10):
    print("Clustor %d" % cluster)
    words = []
    for i in range(0, len(word_centroid_map.values())):
        if word_centroid_map.values()[i] == cluster:
            words.append(word_centroid_map.keys()[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values() + 1)

    bag_of_centroids = np.zeros(num_centroids, dtype='float32')
    for word in wordlist:
        if word in word_centroid_map:
            index = word.centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


################################################################################

train_centroids = np.zeros(
    (train['review'].size, num_clusters), dtype='float32'
)

counter = 0
for review in clean_train_reviews:
    train_centronds[counter] = create_bag_of_centroids(review, word_centroid_map)
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

output = pd.DataFrame(datq={"id":test['id'], "sentiment":result})
output.to_csv("BagOfCentroids.dcdcccccccxcczcsv", index=False, quoting=3)
