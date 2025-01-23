import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK packages
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text_v2(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text by splitting on whitespace
    tokens = text.split()

    # Remove stopwords
    stop_words = ENGLISH_STOP_WORDS
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    # Combine back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Example passage
passage = """
Natural Language Processing is an exciting field of Artificial Intelligence.
It deals with the interaction between computers and humans using natural language.
"""

# Apply preprocessing
preprocessed_passage = preprocess_text_v2(passage)

# Output
print("Original Passage:")
print(passage)
print("\nPreprocessed Passage:")
print(preprocessed_passage)