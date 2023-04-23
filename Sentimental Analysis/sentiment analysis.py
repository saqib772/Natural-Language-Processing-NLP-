# Import the necessary libraries
import nltk
from nltk.corpus import movie_reviews
#Need to make sure First install the nltk library with PIP and then run the model.
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Define a function to extract features from the movie reviews
def extract_features(words):
    return dict([(word, True) for word in words])

# Load the movie reviews dataset
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

# Extract the features from the movie reviews
features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_fileids]

# Split the dataset into training and testing sets
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]

# Train the Naive Bayes classifier
nb_classifier = NaiveBayesClassifier.train(features_train)

# Test the classifier on the testing set
accuracy = nltk_accuracy(nb_classifier, features_test)

print('Accuracy:', accuracy)

# Test the classifier on some sample reviews
input_reviews = [
    'The movie was great!',
    'The movie was not good.',
    'The plot was predictable.',
    'The acting was terrible.',
    'The movie was excellent.',
    'The movie was a waste of time.',
    'The movie was really bad.'
]

for review in input_reviews:
    print('\nReview:', review)
    probdist = nb_classifier.prob_classify(extract_features(review.split()))
    sentiment = probdist.max()
    print('Sentiment:', sentiment)
    print('Probability:', round(probdist.prob(sentiment), 2))
