import numpy as np
import random

def get_coordinates_labels(y_hsi):
    max_label = np.max(y_hsi)
    row_coords = []
    col_coords = []
    labels = []
    for lbl in range(1, max_label+1):
        real_label = lbl - 1
        lbl_locs = np.where(y_hsi == lbl)
        row_coords.append(lbl_locs[0])
        col_coords.append(lbl_locs[1])
        length = len(lbl_locs[0])
        labels.append(np.array([real_label]*length))
    row_coords = np.expand_dims(np.concatenate(row_coords), axis=-1)
    col_coords = np.expand_dims(np.concatenate(col_coords), axis=-1)
    return np.concatenate([row_coords, col_coords], axis=-1), np.concatenate(labels)

def get_train_test(data, data_labels, val_size, test_size = None, shuffle = True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    val_data = []
    val_labels = []
    limited_num = None

    unique_labels = np.unique(data_labels)
    for index, label in enumerate(unique_labels):
        masks = (data_labels == label)
        length = masks.sum()
        if test_size:
            nb_test = int(test_size * length)
            nb_val = int(val_size * length)
            test_indexes = np.random.choice(length, (nb_test,), replace=False)
            test_indexes = test_indexes.tolist()
            val_indexes = random.sample(test_indexes, nb_val)
            train_indexes = np.array([i for i in range(length) if i not in test_indexes])
            val_indexes = np.array(val_indexes)
            test_indexes = np.array(test_indexes)
        if limited_num:
            assert (test_size is None)
            if label==0:
                nb_train=3313
            if label==1:
                nb_train=3919

            train_indexes = np.random.choice(length, (nb_train,), replace=False)
            val_indexes = np.random.choice(length, (1500,), replace=False)
            test_indexes = np.array([i for i in range(length) if i not in train_indexes])
            train_labels.extend(data_labels[masks][train_indexes])
            test_labels.extend(data_labels[masks][test_indexes])
            val_labels.extend(data_labels[masks][val_indexes])
        else:
            train_labels.extend(data_labels[masks][train_indexes])
            test_labels.extend(data_labels[masks][test_indexes])
            val_labels.extend(data_labels[masks][val_indexes])

        train_data.append(data[masks][train_indexes])
        test_data.append(data[masks][test_indexes])
        val_data.append(data[masks][val_indexes])

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.array(test_labels)
    val_data = np.concatenate(val_data, axis=0)
    val_labels = np.array(val_labels)
    if shuffle:
        train_shuffle = np.random.permutation(len(train_labels))
        train_data = train_data[train_shuffle]
        train_labels = train_labels[train_shuffle]
        test_shuffle = np.random.permutation(len(test_labels))
        test_data = test_data[test_shuffle]
        test_labels = test_labels[test_shuffle]
        val_shuffle = np.random.permutation(len(val_labels))
        val_data = val_data[val_shuffle]
        val_labels = val_labels[val_shuffle]
    return train_data, train_labels, test_data, test_labels, val_data, val_labels

def get_train_val_test(data, data_labels, test_size = None, shuffle = True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    limited_num = None
    unique_labels = np.unique(data_labels)
    for index, label in enumerate(unique_labels):
        masks = (data_labels == label)
        length = masks.sum()
        if test_size:
            nb_test = int(test_size * length)
            test_indexes = np.random.choice(length, (nb_test,), replace=False)
            train_indexes = np.array([i for i in range(length) if i not in test_indexes])
        if limited_num:
            assert (test_size is None)
            nb_train = limited_num
            train_indexes = np.random.choice(length, (nb_train,), replace=False)
            test_indexes = np.array([i for i in range(length) if i not in train_indexes])
        else:
            train_labels.extend(data_labels[masks][train_indexes])
            test_labels.extend(data_labels[masks][test_indexes])
        train_data.append(data[masks][train_indexes])
        test_data.append(data[masks][test_indexes])
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.array(test_labels)
    if shuffle:
        train_shuffle = np.random.permutation(len(train_labels))
        train_data = train_data[train_shuffle]
        train_labels = train_labels[train_shuffle]
        test_shuffle = np.random.permutation(len(test_labels))
        test_data = test_data[test_shuffle]
        test_labels = test_labels[test_shuffle]
    return train_data, train_labels, test_data, test_labels