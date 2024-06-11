# RNN code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def RNN(
    word_index,
    embedding_matrix,
    max_seq_length = 100, # original 1000
    embedding_dim = 100, # embedding dimension (GloVe size)
):
    model = Sequential([
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],input_length=max_seq_length, trainable=False),
        LSTM(100, return_sequences=True),
        Dropout(0.4),
        LSTM(100),
        Dense(20, activation='softmax')
    ])
    return model