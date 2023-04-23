import tensorflow as tf
import numpy as np

input_language = 'en' # English
output_language = 'es' # Spanish
!wget http://storage.googleapis.com/download.tensorflow.org/data/eng-spa.zip
!unzip ted_hrlr_translate/eng-spa.zip

dataset = tf.keras.utils.get_file(fname="ted_hrlr_translate/eng-spa.zip",origin="http://storage.googleapis.com/download.tensorflow.org/data/eng-spa.zip",extract=True)

tokenizer_input = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer_output = tf.keras.preprocessing.text.Tokenizer(filters='')

def tokenize(sentences):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    return sequences, tokenizer

input_sentences = []
output_sentences = []

with open('eng-spa.txt', 'r', encoding='utf8') as f:
    lines = f.read().split('\n')
    for line in lines[:len(lines)-1]:
        input_sentence, output_sentence, _ = line.split('\t')
        input_sentences.append(input_sentence)
        output_sentences.append(output_sentence)

input_sequences, tokenizer_input = tokenize(input_sentences)
output_sequences, tokenizer_output = tokenize(output_sentences)

latent_dim = 256

encoder_inputs = tf.keras.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(len(tokenizer_input.word_index)+1, latent_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(len(tokenizer_output.word_index)+1, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
attention = tf.keras.layers.Attention()([decoder_outputs, encoder_outputs])
decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention])
decoder_dense = tf.keras.layers.Dense(len(tokenizer_output.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 64
epochs = 50

model.fit([input_sequences, output_sequences[:,:-1]], output_sequences[:,1:],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


def translate_sentence(input_sentence):
    encoder_input = tokenizer_input.texts_to_sequences([input_sentence])
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(encoder_input, maxlen=max_input_length, padding='post')
    decoder_input = np.zeros((1, max_output_length))
    decoder_input[0, 0] = tokenizer_output.word_index['<start>']
    for i in range(1, max_output_length):
        predictions = model.predict([encoder_input, decoder_input])
        token_index = np.argmax(predictions[0,i-1,:])
        if token_index == tokenizer_output.word_index['<end>']:
            break
        decoder_input[0, i] = token_index
    translated_sentence = ''
    for token in decoder_input[0]:
        if token != 0 and token != tokenizer_output.word_index['<start>'] and token != tokenizer_output.word_index['<end>']:
            translated_sentence += tokenizer_output.index_word[token] + ' '
    return translated_sentence
