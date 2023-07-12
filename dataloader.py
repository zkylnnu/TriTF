import numpy as np
from numba import jit
def simple_data_generator(X1, X2, X3, y, batch_size = 24, shuffle = True, till_end = False):

    data_length = len(y)
    indexes = np.array(list(range(data_length)))
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi*batch_size:(epi+1)*batch_size]
        X_batch1 = X1[selected]
        X_batch2 = X2[selected]
        X_batch_slic = X3[selected]
        X_batch3_sm = notlocal_aru(X_batch_slic, batch_size)
        y_batch = y[selected]
        yield X_batch1, X_batch2, X_batch_slic, y_batch, X_batch3_sm
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch1 = X1[selected]
            X_batch2 = X2[selected]
            y_batch = y[selected]
            yield X_batch1, X_batch2, y_batch

def simple_data_generator_test(X1, X2, X3, y, realcoords, batch_size = 24, shuffle = True, till_end = False):

    data_length = len(y)
    indexes = np.array(list(range(data_length)))
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi*batch_size:(epi+1)*batch_size]
        X_batch1 = X1[selected]
        X_batch2 = X2[selected]
        X_batch_slic = X3[selected]
        X_batch3_sm = notlocal_aru(X_batch_slic, batch_size)
        y_batch = y[selected]
        order = realcoords[selected]
        yield X_batch1, X_batch2, X_batch_slic, y_batch, X_batch3_sm, order
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch1 = X1[selected]
            X_batch2 = X2[selected]
            y_batch = y[selected]
            yield X_batch1, X_batch2, y_batch
@jit
def notlocal_aru(X_batch,batchsize):
    X_batcha = X_batch
    X_1c = np.mean(X_batcha,axis=2)
    X_1c = np.reshape(X_1c,[batchsize,-1])
    X_similar = np.zeros_like(X_batch)
    best_list = np.zeros(batchsize)
    for i in range(batchsize):
        patch1 = X_1c[i]
        lowdiss = 10e6
        best_sm = 0
        for j in range(batchsize):
            if i==j:
                continue
            patch2 = X_1c[j]
            diff = patch1 - patch2
            diss = np.multiply(diff,diff)
            total_diss = np.sum(diss)
            if total_diss<lowdiss:
                lowdiss = total_diss
                best_sm = j
        best_list[i] = best_sm
        X_similar[i] = X_batch[best_sm]
    return best_list


