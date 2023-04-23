# Sentimental Analysis Project 

we first import the necessary libraries, including NLTK.

You can install the nltk library using below code:
```
pip install nltk
```
We then define a function called extract_features that extracts features from a list of words. This function returns a dictionary where each word is a key and the value is True.

Next, we load the movie reviews dataset from NLTK and extract the features from the positive and negative reviews. We then split the dataset into training and testing sets and train a Naive Bayes classifier using the training set.

We then test the classifier on the testing set and calculate its accuracy. 

Finally, we test the classifier on some sample reviews and print the predicted sentiment and probability for each review.

This code should give you a basic understanding of how to build a sentiment analysis model using NLTK. However, note that this is a simple example and there are many ways to improve the accuracy of the model, such as using more advanced feature extraction techniques and trying different classifiers.

For more Advanced Versions You Can Check the Second File in This Repository as Sentimental_Analysis_Updated.
