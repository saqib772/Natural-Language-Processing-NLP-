import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest

# Sample Text
text = "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is a rapidly growing field, with many applications in various industries. However, one of the challenges in AI is to develop systems that can understand human language and generate human-like responses."

# Tokenize sentences
sentences = sent_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
words = nltk.word_tokenize(text)
words = [word for word in words if word.casefold() not in stop_words]

# Calculate word frequency
word_frequencies = {}
for word in words:
    if word not in word_frequencies:
        word_frequencies[word] = 1
    else:
        word_frequencies[word] += 1

# Calculate sentence scores based on word frequency
sentence_scores = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word_frequencies:
            if len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

# Get top sentences
summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)


The Answer is " 
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest

# Sample Text
text = "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is a rapidly growing field, with many applications in various industries. However, one of the challenges in AI is to develop systems that can understand human language and generate human-like responses."

# Tokenize sentences
sentences = sent_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
words = nltk.word_tokenize(text)
words = [word for word in words if word.casefold() not in stop_words]

# Calculate word frequency
word_frequencies = {}
for word in words:
    if word not in word_frequencies:
        word_frequencies[word] = 1
    else:
        word_frequencies[word] += 1

# Calculate sentence scores based on word frequency
sentence_scores = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word_frequencies:
            if len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

# Get top sentences
summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)

#The answe is 
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest

# Sample Text
text = "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI is a rapidly growing field, with many applications in various industries. However, one of the challenges in AI is to develop systems that can understand human language and generate human-like responses."

# Tokenize sentences
sentences = sent_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
words = nltk.word_tokenize(text)
words = [word for word in words if word.casefold() not in stop_words]

# Calculate word frequency
word_frequencies = {}
for word in words:
    if word not in word_frequencies:
        word_frequencies[word] = 1
    else:
        word_frequencies[word] += 1

# Calculate sentence scores based on word frequency
sentence_scores = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word_frequencies:
            if len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

# Get top sentences
summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)

#The answer is  "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. However, one of the challenges in AI is to develop systems that can understand human language and generate human-like responses.
