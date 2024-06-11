# CNN code
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as st

def cnn(
    word_index,
    embedding_matrix,
    max_seq_length = 100, # original 1000
    embedding_dim = 100, # embedding dimension (GloVe size)
    filter_sizes = [5,5,5],
    num_filters = 512,
    drop = 0.5
):
    inputs = Input(shape=(max_seq_length,), dtype = 'int32')
    
    embedding = Embedding(
        len(word_index) + 1,
        embedding_dim,
        weights = [embedding_matrix],
        input_length = max_seq_length,
        trainable = False
    )(inputs)

    reshape = Reshape((max_seq_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size = (filter_sizes[0], embedding_dim), padding ='valid', kernel_initializer = 'normal', activation = 'relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size = (filter_sizes[1], embedding_dim), padding ='valid', kernel_initializer = 'normal', activation = 'relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size = (filter_sizes[2], embedding_dim), padding ='valid', kernel_initializer = 'normal', activation = 'relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size = (max_seq_length - filter_sizes[0] + 1, 1), strides = (1,1), padding = 'valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size = (max_seq_length - filter_sizes[1] + 1, 1), strides = (1,1), padding = 'valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size = (max_seq_length - filter_sizes[2] + 1, 1), strides = (1,1), padding = 'valid')(conv_2)

    concatenated_tensor = Concatenate(axis = 1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units = 20, activation = 'softmax')(dropout)

    model = Model(inputs = inputs, outputs = output)
    
    return model

def ensemble(
    embeddings_index,
    y_test
):
    m = {
        0: 'atheism',
        1: 'computers graphics',
        2: 'computer operating system microsoft windows miscellaneous',
        3: 'computer system pc hardware',
        4: 'computer system mac hardware',
        5: 'computer windows',
        6: 'miscellaneous for sale',
        7: 'recreational autos',
        8: 'recreational motorcycles',
        9: 'recreational sport baseball',
        10: 'recreational sport hockey',
        11: 'science cryptography',
        12: 'science electronics',
        13: 'science medicine',
        14: 'science space',
        15: 'social religion christian',
        16: 'talk politics guns',
        17: 'talk politics mideast',
        18: 'talk politics miscellaneous',
        19: 'talk religion miscellaneous'
    }
    models = ['LR', 'LSVC', 'SVC', 'SGD', 'RNN', 'CNN']
    preds = [np.load(f'{i}_pred.npy') for i in models] # load predictions from models
    preds_c = [[(i, m[i]) for i in p] for p in preds] # map predictions back to text
    embedded_c = np.array([np.average([embeddings_index[j] for j in v.split()], axis = 0) for v in m.values()]) # embed the categories using GloVe average for multiword
    counts = np.array([799, 973, 985, 982, 963, 988, 975, 990, 996, 994, 999, 991, 984, 990, 987, 997, 910, 940, 775, 628])
    weights = (counts / sum(counts)).reshape(20,1)
    embedded_c_w = embedded_c * weights
    preds_embed = np.array([[embedded_c[i] for i,p in model] for model in preds_c]) # map prediction texts into their average embeddings
    preds_embed_w = np.array([[embedded_c_w[i] for i,p in model] for model in preds_c]) # map prediction texts into their weighted average embeddings

    for t,e in [('average embedding', preds_embed), ('average (weighted)', preds_embed_w)]:
        for i,m1 in enumerate(e):
            for j,m2 in enumerate(e[i+1:], start = i +1):
                av_ps = np.array([np.average([p1, p2], axis = 0) for p1,p2 in zip(m1, m2)]) # averaging the embedding prediction vectors of both models
                closest_cats = np.array([np.argmax([cosine_similarity([p], [cat])[0][0] for cat in embedded_c]) for p in av_ps]) # finding closest category
                accuracy = np.mean(accuracy_score(np.argmax(y_test, axis = 1).flatten(), closest_cats.flatten())) # accuracy calculation
                print(f'{models[i]} + {models[j]} give an {t} accuracy of: ', accuracy)

    # mode
    mode = np.array(st.mode(np.array(preds[:4]), keepdims = False))[0]
    mode_accuracy = np.mean(accuracy_score(np.argmax(y_test, axis = 1).flatten(), mode)) # accuracy calculation
    print('Taking the mode of LR, SVC and SGD predictions we achieve an accuracy of: ', mode_accuracy)