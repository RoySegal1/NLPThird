import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter

from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
import tensorflow as tf

def print_word_statistics(words, title):
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    most_common_words = word_counts.most_common(5)

    print(f"{title} Statistics:")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Most common words: {most_common_words}")
    print("\n")

# Function to tokenize text
def tokenize(text):
    return word_tokenize(text)

# Function to lemmatize tokens
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to remove stopwords and punctuation, and convert to lowercase
def preprocess_text(tokens):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [token.lower() for token in tokens if token.lower() not in stopwords and token.isalpha()]

# # URLs of Wikipedia pages
# url_turing = 'https://en.wikipedia.org/wiki/Alan_Turing'
# url_einstein = 'https://en.wikipedia.org/wiki/Albert_Einstein'
#
# # Function to scrape text from Wikipedia pages
# def scrape_wikipedia(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     paragraphs = soup.find_all('p')
#     data = [paragraph.text for paragraph in paragraphs]
#     return ' '.join(data)

# Scrape text from Wikipedia pages


#data_einstein = scrape_wikipedia(url_einstein)
with open('data_turing.txt', 'r', encoding='utf-8') as file:
    data_turing = file.read()

sentences_turing = sent_tokenize(data_turing)
words_turing_tokenized = [preprocess_text(word_tokenize(sentence)) for sentence in sentences_turing]

# Tokenize, preprocess, and lemmatize text
tokens_turing = tokenize(data_turing)


filtered_tokens_turing = preprocess_text(tokens_turing)


lemmas_turing = lemmatize(filtered_tokens_turing)


# Calculate Bag-of-Words (FreqDist) for Alan Turing
bow_turing = Counter(lemmas_turing)
most_common_words = bow_turing.most_common(5)
print("Bag-of-Words (FreqDist) for Alan Turing:\n", most_common_words)
print("\n")



# Train Word2Vec model
model = Word2Vec(words_turing_tokenized, vector_size=10, window=5, workers=4)
word = 'turing'
# Example usage of the trained model
if word in model.wv.key_to_index:
    vector = model.wv[word]
    print(f"Vector for '{word}':", vector)
else:
    print(f"'{word}' not found in the vocabulary")

similar_words = model.wv.most_similar('work')  # Get most similar words to "machine"

# print("Vector for 'machine':", vector)
print("Most similar words to 'work':", similar_words)

# Create sequences of words for the RNN model
sequence_length = 5
sequences = []
for i in range(sequence_length, len(lemmas_turing)):
    seq = lemmas_turing[i-sequence_length:i+1]
    sequences.append(seq)

# Create the vocabulary and map words to integers
vocab = sorted(set(lemmas_turing))
word_to_int = {word: idx for idx, word in enumerate(vocab)}
int_to_word = {idx: word for idx, word in enumerate(vocab)}

# Convert sequences to integer sequences
sequences_int = [[word_to_int[word] for word in seq] for seq in sequences]

# Prepare data for the RNN model
X = np.array([seq[:-1] for seq in sequences_int])
y = np.array([seq[-1] for seq in sequences_int])
y = tf.keras.utils.to_categorical(y, num_classes=len(vocab))

# Build the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=50, input_length=sequence_length),
    tf.keras.layers.SimpleRNN(100, return_sequences=False),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64)

# Function to predict the next word
def predict_next_word(model, sequence):
    sequence = [word_to_int[word] for word in sequence]
    sequence = np.array(sequence).reshape(1, -1)
    prediction = model.predict(sequence, verbose=0)
    next_word = int_to_word[np.argmax(prediction)]
    return next_word

# Example usage
test_sequence = lemmas_turing[:sequence_length]
next_word = predict_next_word(model, test_sequence)
print(f"Next word prediction: {next_word}")