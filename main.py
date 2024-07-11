import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import string
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
# Function to tokenize text
stopwords = set(nltk.corpus.stopwords.words('english'))


# Function to lemmatize tokens
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# Function to remove stopwords and punctuation, and convert to lowercase
def preprocess_text(tokens):
    return [token.lower() for token in tokens if token.lower() not in stopwords and token.isalpha()]


# Define text preprocessing function
def preprocess_text2(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Load text data
with open('data_turing.txt', 'r', encoding='utf-8') as file:
    data_turing = file.read()

# Tokenize, preprocess, and lemmatize text
sentences_turing = sent_tokenize(data_turing)
words_turing_tokenized = [preprocess_text(word_tokenize(sentence)) for sentence in sentences_turing]
sentences_turing_preprocessed = [preprocess_text2(sentence) for sentence in sentences_turing]
tokens_turing = word_tokenize(data_turing)
filtered_tokens_turing = preprocess_text(tokens_turing)
lemmas_turing = lemmatize(filtered_tokens_turing)

# Calculate Bag-of-Words (FreqDist) for Alan Turing
bow_turing = Counter(lemmas_turing)
most_common_words = bow_turing.most_common(5)
print("Bag-of-Words (FreqDist) for Alan Turing:\n", most_common_words)
print("\n")

# Train Word2Vec model
embedding_dim = 10
w2v_model = Word2Vec(words_turing_tokenized, vector_size=embedding_dim, window=5, workers=4)
word = 'turing'

# Example usage of the trained model
if word in w2v_model.wv.key_to_index:
    vector = w2v_model.wv[word]
    print(f"Vector for '{word}':", vector)
else:
    print(f"'{word}' not found in the vocabulary")

similar_words = w2v_model.wv.most_similar('work')
print("Most similar words to 'work':", similar_words)

# Create sequences of words for the RNN model
sequence_length = 8
sequences = []
for sentence in words_turing_tokenized:
    for i in range(sequence_length, len(sentence)):
        seq = sentence[i - sequence_length:i + 1]
        sequences.append(seq)

# Create the vocabulary and map words to integers
vocab = sorted(set(filtered_tokens_turing))
word_to_int = {word: idx for idx, word in enumerate(vocab)}
int_to_word = {idx: word for idx, word in enumerate(vocab)}

# Convert sequences to integer sequences
sequences_int = [[word_to_int[word] for word in seq] for seq in sequences]

# Prepare data for the RNN model
X = np.array([seq[:-1] for seq in sequences_int])
y = np.array([seq[-1] for seq in sequences_int])
y = tf.keras.utils.to_categorical(y, num_classes=len(vocab))

# Extract embeddings from Word2Vec model
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, idx in word_to_int.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    else:
        embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))

# Build the RNN model with the pre-trained embeddings
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim, weights=[embedding_matrix],
                              input_length=sequence_length, trainable=False),
    tf.keras.layers.SimpleRNN(100, return_sequences=False),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the LSTM model with the pre-trained embeddings
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=embedding_dim, weights=[embedding_matrix],
                              input_length=sequence_length, trainable=False),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the RNN model
rnn_history = rnn_model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

# Train the LSTM model
lstm_history = lstm_model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)


# Function to predict the next word using the RNN model
def predict_next_word_rnn(model, sequence):
    sequence = [word_to_int[word] for word in sequence]
    sequence = np.array(sequence).reshape(1, -1)
    prediction = model.predict(sequence, verbose=0)
    next_word = int_to_word[np.argmax(prediction)]
    return next_word


# Function to predict the next word using the LSTM model
def predict_next_word_lstm(model, sequence):
    sequence = [word_to_int[word] for word in sequence]
    sequence = np.array(sequence).reshape(1, -1)
    prediction = model.predict(sequence, verbose=0)
    next_word = int_to_word[np.argmax(prediction)]
    return next_word


# Example usage
test_sequence = words_turing_tokenized[2][:8]
print(f"Test sequence: {test_sequence}")

# Predict next word using RNN
next_word_rnn = predict_next_word_rnn(rnn_model, test_sequence)
print(f"RNN next word prediction: {next_word_rnn}")

# Predict next word using LSTM
next_word_lstm = predict_next_word_lstm(lstm_model, test_sequence)
print(f"LSTM next word prediction: {next_word_lstm}")
# Compare performance
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['accuracy'])
plt.plot(rnn_history.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['accuracy'])
plt.plot(lstm_history.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['loss'])
plt.plot(rnn_history.history['val_loss'])
plt.title('RNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
#We can see from the graphs that their is high variance,
# and very low bias which can indicate over fitting ,
# we assume that is the case because of the small corpus size.


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are available
# nltk.download('punkt')

# Load text data
with open('data_turing.txt', 'r', encoding='utf-8') as file:
    data_turing = file.read()

# Select 5 partial sentences from your corpus
sentences_turing = sent_tokenize(data_turing)
partial_sentences = [
    "He was highly influential in the development of",
    "Turing devised techniques",
    "Despite these accomplishments, he was never fully",
    "After graduating from Sherborne",
    "Turing spent most of his time studying"
]
# Find original complete sentences
original_sentences = [s for s in sentences_turing if any(ps in s for ps in partial_sentences)]

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # or your fine-tuned model path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate completions for each partial sentence
for i, partial_sentence in enumerate(partial_sentences):
    # Encode the input text
    input_ids = tokenizer.encode(partial_sentence, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode and print the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Original: {partial_sentence}\nGenerated: {generated_text}\n")


sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Perform sentiment analysis on each sentence
sentiments = sentiment_pipeline(sentences_turing)

# Initialize counters
very_positive_count = 0
positive_count = 0
neutral_count = 0
negative_count = 0
very_negative_count = 0

very_positive_examples = []
positive_examples = []
neutral_examples = []
negative_examples = []
very_negative_examples = []

# Count sentiments and store examples
for sentence, sentiment in zip(sentences_turing, sentiments):
    label = sentiment['label']
    if label == '1 star':
        very_negative_count += 1
        very_negative_examples.append(sentence)
    elif label == '2 stars':
        negative_count += 1
        negative_examples.append(sentence)
    elif label == '3 stars':
        neutral_count += 1
        neutral_examples.append(sentence)
    elif label == '4 stars':
        positive_count += 1
        positive_examples.append(sentence)
    elif label == '5 stars':
        very_positive_count += 1
        very_positive_examples.append(sentence)

# Calculate percentages
total_sentences = len(sentences_turing)
very_positive_percentage = (very_positive_count / total_sentences) * 100
positive_percentage = (positive_count / total_sentences) * 100
neutral_percentage = (neutral_count / total_sentences) * 100
negative_percentage = (negative_count / total_sentences) * 100
very_negative_percentage = (very_negative_count / total_sentences) * 100

# Print the results
print(f"Total Sentences: {total_sentences}")
print(f"Very Positive Sentences: {very_positive_percentage:.2f}%")
print(f"Positive Sentences: {positive_percentage:.2f}%")
print(f"Neutral Sentences: {neutral_percentage:.2f}%")
print(f"Negative Sentences: {negative_percentage:.2f}%")
print(f"Very Negative Sentences: {very_negative_percentage:.2f}%")


# Print some examples for each category
def print_examples(label, examples):
    print(f"\nExamples of {label} sentences:")
    for example in examples[:3]:  # Print up to 3 examples
        print(f"- {example}")


print_examples("Very Positive", very_positive_examples)
print_examples("Positive", positive_examples)
print_examples("Neutral", neutral_examples)
print_examples("Negative", negative_examples)
print_examples("Very Negative", very_negative_examples)
