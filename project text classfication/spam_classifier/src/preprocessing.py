import string
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def create_dictionary(messages):
    dictionary = {}
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = len(dictionary)
    return dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary[token]] += 1
    return features

def build_feature_matrix(messages, dictionary):
    return np.array([create_features(tokens, dictionary) for tokens in messages])
