from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# occurs after the preprocessing and the split 
tokenizer = Tokenizer()
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

def lsvc(x_train_tfidf, x_val_tfidf, y_train,  y_val):
    # function to run the LinearSVC model
    def lsvc_box_function(C):
        # C: SVC hyper parameter to optimize for.
        model = LinearSVC(C = C)
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_val_tfidf)
        f = accuracy_score(y_val, y_pred)
        return f

    # Bounded region of parameter space
    pbounds = {"C": [0.01, 10]}
    optimizer = BayesianOptimization(
        f=lsvc_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )    
    return LinearSVC(C = optimizer.max['params']['C'])

def svc(x_train, x_test, y_train, y_test):
    # Apply TFIDF on training and validation 
    x_train_vectors_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_vectors_tfidf = tfidf_vectorizer.transform(x_test)
    
    # function to run the LinearSVC model
    def svc_box_function(C):
        # C: SVC hyper parameter to optimize for.
        model = SVC(C = C, kernel="linear")
        model.fit(x_train_vectors_tfidf, y_train)
        y_score = model.predict(x_test_vectors_tfidf)
        f = accuracy_score(y_test, y_score)
        return f
    
    # Bounded region of parameter space
    pbounds = {"C": [0.01, 10]}
    optimizer = BayesianOptimization(
        f=svc_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )
    return SVC(C = optimizer.max['params']['C'])