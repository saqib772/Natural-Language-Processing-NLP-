# Language Translation using Deep Learning

This code implements a language translation module using deep learning techniques. Specifically, we use a sequence-to-sequence model with an encoder and decoder architecture to translate sentences from one language to another. We train the model on a dataset of parallel sentences in the two languages.

# Prerequisites

To run this code, you will need the following libraries:

1. Tensorflow (version 2.0 or higher)
2. Numpy
3. Pandas
4. Scikit-learn

Install These Libraries: 
``` pip install tensorflow numpy pandas scikit-learn```

# Dataset

The dataset used for training and testing the model is a parallel corpus of English and French sentences, available from the Tatoeba Project.


# Data Preprocessing

The first step in preparing the data for training is to tokenize the sentences and convert them into sequences of integer indices. We use the Tokenizer class from the tensorflow.keras.preprocessing.text module to perform this task. We fit the tokenizer on the training data and then use it to transform both the training and testing data.

Next, we pad the sequences to a fixed length using the pad_sequences function from the same module. This ensures that all sequences have the same length, which is necessary for efficient training of the model.

# Model ArchitectureThe translate_sentence function takes an input sentence in the source language and generates a translated sentence in the target language using the trained model. The function first tokenizes and pads the input sentence, and then initializes the decoder input with a start token. It then runs the model in a loop to generate the output sentence one word at a time, until it reaches the maximum output length or predicts an end token.

Finally, the function converts the output sequence back into a sentence and returns it.
The model architecture consists of an encoder and decoder, both implemented as recurrent neural networks (RNNs) with long short-term memory (LSTM) cells. The encoder processes the input sentence and generates a context vector, which is passed to the decoder to generate the output sentence.

The output sentence is generated one word at a time, with the decoder using the context vector and the previous generated word to predict the next word in the sequence. We use teacher forcing during training, where the decoder inputs are the ground truth output words, rather than the predicted words from the previous time step.

The model is trained using the Adam optimizer and categorical cross-entropy loss function. We use early stopping and model checkpointing to prevent overfitting and save the best performing model.

# Translation Function

The translate_sentence function takes an input sentence in the source language and generates a translated sentence in the target language using the trained model. The function first tokenizes and pads the input sentence, and then initializes the decoder input with a start token. It then runs the model in a loop to generate the output sentence one word at a time, until it reaches the maximum output length or predicts an end token.

Finally, the function converts the output sequence back into a sentence and returns it.
