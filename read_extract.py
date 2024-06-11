import os
import numpy as np
import pandas as pd
import nltk
import regex as re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def remove_stopwords(text):
    t = word_tokenize(text)
    s = stopwords.words('english')
    return ' '.join([w for w in t if w.lower() not in s])

def read_file():
    d = pd.read_csv('data_c.csv') # data csv with 'text' and 'label' columns
    return d['text'], d['label']

def tvs(x, y, t = 0.2, v = 0.2, r = 0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = t, random_state = r)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = v / (1-t), random_state = r)

    return x_train, x_val, x_test, y_train, y_val, y_test

def fe_tfidf_svd(total_texts, y_target):
    # tf-idf matrix
    vectorizer = TfidfVectorizer(stop_words = 'english')
    X_tfidf = vectorizer.fit_transform(total_texts)
    features = vectorizer.get_feature_names_out()
    
    # SVD applied to tf-idf
    n_components = 500
    svd = TruncatedSVD(n_components=n_components) # Initialize TruncatedSVD with n_components
    X_reduced = svd.fit_transform(X_tfidf) # Fit and transform the TF-IDF matrix
    y_target_array = np.array(y_target)
    
    # term docuemnt matrix
    termdoc = CountVectorizer().fit_transform(total_texts)

    return X_tfidf, features

def fe_keywords(x_tfidf, features, top_n):
    top_n_features = []
    for r in range(x_tfidf.shape[0]):  # Iterate through each document/row
        row = np.squeeze(x_tfidf[r].toarray())  # Convert the row to a dense format
        top_n_ids = np.argsort(row)[::-1][:top_n]  # Get indices of top features
        top_feats = [(features[i], row[i]) for i in top_n_ids]  # Extract feature names and scores
        top_n_features.append(top_feats)
    return top_n_features

def hs_preprocessing(text, subject_weight=3): # Weight is the number of times the subject will be repeated
    # Isolating the subject line and removing the "Subject:" prefix
    subject = re.search(r'^Subject:.*', text, flags=re.MULTILINE)
    subject_line = ''
    if subject:
        subject_line = subject.group().replace('Subject:', '').strip()
        # Remove the subject line from the text
        text = text.replace(subject.group(), '')

    # Removes other headers
    text = re.sub(r'^.*:.*(?:\n|\r\n?)', '', text, flags=re.MULTILINE)

    # Excludes uuencoded text that usually starts from "begin XXX" or "------------ Part" and onwards 
    text = re.split(r'^begin \d{3}|^------------ Part', text, flags=re.MULTILINE, maxsplit=1)[0]

    # Removes email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', ' ', text)

    # Converts to lowercase
    text = text.lower()

    # Repeats the subject line to give it more weight
    weighted_subject = ' '.join([subject_line.lower()] * subject_weight)

    # Combines the weighted subject with the body
    text = weighted_subject + ' ' + text

    # Removes special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenizes
    tokens = word_tokenize(text)

    # Removes stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Rejoin tokens into a single string
    return ' '.join(tokens)

