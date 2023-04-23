# Import the necessary libraries
import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Load the movie reviews dataset
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

# Create a list of documents and their corresponding labels
documents = [(list(movie_reviews.words(fileid)), category)
             for category in ['pos', 'neg']
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure randomness
import random
random.seed(42)
random.shuffle(documents)

# Split the dataset into training and testing sets
train_documents = [(" ".join(review_words), category) for (review_words, category) in documents[:int(len(documents) * 0.8)]]

test_documents = documents[int(len(documents) * 0.8):]

# Extract features from the training set using TF-IDF vectorization
def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words) if words else " "

vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 2), stop_words=None, preprocessor=remove_stopwords)
train_features = vectorizer.fit_transform([" ".join(d[0]) for d in train_documents if d[0]])

# Train and evaluate different classifiers
from sklearn.multiclass import OneVsRestClassifier

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Linear SVM': LinearSVC(max_iter=10000)
}

for name, classifier in classifiers.items():
    # Wrap the classifier in a OneVsRestClassifier
    classifier = OneVsRestClassifier(SklearnClassifier(classifier))
    
    # Define the pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train the pipeline
    pipeline.fit([d[0] for d in train_documents], [d[1] for d in train_documents])
    
    # Make predictions on the testing set and evaluate the accuracy
    predicted_labels = pipeline.predict([d[0] for d in test_documents])
    true_labels = [d[1] for d in test_documents]
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f'{name} accuracy:', accuracy)
