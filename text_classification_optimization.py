from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from scipy.sparse import vstack
import numpy as np

def svc(x_train_tfidf, x_val_tfidf, y_train, y_val):
    
    # Function to run the LinearSVC model
    def lsvc_box_function(C):
        # C: LinearSVC hyperparameter to optimize for.
        model = LinearSVC(C=C, max_iter=1000)  # Set max_iter to ensure convergence
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_val_tfidf)
        return accuracy_score(y_val, y_pred)

    # Bounded region of parameter space
    pbounds = {"C": (0.01, 10)}

    optimizer = BayesianOptimization(
        f=lsvc_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )

    best_C = optimizer.max['params']['C']
    # Now, we initialize and return the LinearSVC with the best C found
    best_lsvc = LinearSVC(C=best_C, max_iter=1000)
    best_lsvc.fit(vstack([x_train_tfidf, x_val_tfidf]), np.concatenate([y_train, y_val]))

    return best_lsvc

def lr(x_train_tfidf, x_val_tfidf, y_train, y_val):
    
    # Function to run the Logistic Regression model
    def lr_box_function(C):
        # C: LR hyperparameter to optimize for.
        model = LogisticRegression(C=C, max_iter=1000)  # Set max_iter to ensure convergence
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_val_tfidf)
        return accuracy_score(y_val, y_pred)
    
    # Bounded region of parameter space
    pbounds = {"C": (0.01, 10)}

    optimizer = BayesianOptimization(
        f=lr_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )

    best_C = optimizer.max['params']['C']
    # Now, we initialize and return the Logistic Regression with the best C found
    best_lr = LogisticRegression(C=best_C, max_iter=1000)
    best_lr.fit(vstack([x_train_tfidf, x_val_tfidf]), np.concatenate([y_train, y_val]))

    return best_lr

def sgd(x_train_tfidf, x_val_tfidf, y_train, y_val):
    # Function to run the SGD Classifier model
    def sgd_box_function(alpha):
        # alpha: SGD regularization hyperparameter to optimize for.
        model = SGDClassifier(alpha=alpha, max_iter=1000)  # Set max_iter to a higher value to ensure convergence
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_val_tfidf)
        f = accuracy_score(y_val, y_pred)
        return f
    
    # Bounded region of parameter space
    pbounds = {"alpha": (1e-6, 1e-1)}
    optimizer = BayesianOptimization(
        f=sgd_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=10,
        n_iter=30,
    )
    best_alpha = optimizer.max['params']['alpha']
    # Now, we initialize and return the SGDClassifier with the best alpha found
    best_sgd = SGDClassifier(alpha=best_alpha, max_iter=1000)
    best_sgd.fit(vstack([x_train_tfidf, x_val_tfidf]), np.concatenate([y_train, y_val]))
    
    return best_sgd