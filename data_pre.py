import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import random

def standartsize(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    newX = preprocessing.scale(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX
def standartsize2(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    minMax = preprocessing.StandardScaler()
    newX = minMax.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX

def padWithZeros(X, margin=2):
    return np.pad(X, [(margin, margin), (margin, margin), (0,0)], mode="constant")

def load_dataset(Dataset):
    if Dataset== "FM":
        data_b = sio.loadmat('datasets/farm/farm06.mat')
        data_a = sio.loadmat('datasets/farm/farm07.mat')
        data_before = data_b['imgh']
        data_after = data_a['imghl']
        gt_mat = sio.loadmat('datasets/farm/farm_gt_nonzero.mat')
        gt = gt_mat['farm_gt']
        dataset_name = "farm"
    if Dataset== "RV":
        data_b = sio.loadmat('datasets/river/river_before.mat')
        data_a = sio.loadmat('datasets/river/river_after.mat')
        data_before = data_b['river_before']
        data_after = data_a['river_after']
        gt_mat = sio.loadmat('datasets/river/river_gt.mat')
        gt = gt_mat['river_gt']
        dataset_name = "river"
        gt = 2 - gt
    if Dataset== "USA":
        data_b = sio.loadmat('datasets/USA/USA1.mat')
        data_a = sio.loadmat('datasets/USA/USA2.mat')
        data_before = data_b['USA1']
        data_after = data_a['USA2']
        gt_mat = sio.loadmat('datasets/USA/USA_gt_2c_nonzero.mat')
        gt = gt_mat['USA_gt_2c_nonzero']
        dataset_name = "USA"

    print(dataset_name)
    data_abs = data_before - data_after

    margin = 4
    X1 = padWithZeros(data_before, margin=margin)
    X2 = padWithZeros(data_after, margin=margin)
    X3 = padWithZeros(data_abs, margin=margin)
    return X1, X2, gt, dataset_name, X3

def select_rect_no_pad(hsi1, hsi2, hsi3, target_coords, pitch_size = 7):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    margin = int((pitch_size - 1) / 2)

    pitches1 = [hsi1[target_coords[i,0] - margin:target_coords[i,0] + margin + 1, target_coords[i,1]-margin
                                                   :target_coords[i,1] + margin + 1] for i in range(len(target_coords))]
    pitches2 = [hsi2[target_coords[i, 0] - margin:target_coords[i, 0] + margin + 1, target_coords[i, 1] - margin
                                                                                    :target_coords[i, 1] + margin + 1]
                for i in range(len(target_coords))]
    pitches3 = [hsi3[target_coords[i, 0] - margin:target_coords[i, 0] + margin + 1, target_coords[i, 1] - margin
                                                                                    :target_coords[i, 1] + margin + 1]
                for i in range(len(target_coords))]

    pitches1 = [np.expand_dims(pitch, axis=0) for pitch in pitches1]
    pitches2 = [np.expand_dims(pitch, axis=0) for pitch in pitches2]
    pitches3 = [np.expand_dims(pitch, axis=0) for pitch in pitches3]
    return np.concatenate(pitches1, axis=0), np.concatenate(pitches2, axis=0), np.concatenate(pitches3, axis=0)

def Grammar(img1, img2, tgt_coords, X3, method = "rect 7"):
    try:
        region_type , param = method.split()
        param = int(param)
    except:
        region_type = method
    if region_type == "rect":
        assert (param is not None)
        data1, data2, data_slic = select_rect_no_pad(img1, img2, X3, tgt_coords,pitch_size=param)

    elif region_type == "dot":
        data = select_rect_no_pad(img1, tgt_coords, pitch_size = 1)
        data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2], data.shape[3]])
    return data1, data2, data_slic

def zeropad_to_max_len(data, max_len = 49):
    return np.pad(data, [(0, 0), (0, max_len - data.shape[1]), (0,0)], mode="constant")

def sampling(sampling_mode, train_rate, gt):

    train_rand_idx = []
    gt_1d = np.reshape(gt, [-1])

    if sampling_mode == 'random':

        idx = np.where(gt_1d < 3)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_rate).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_index.append(a[j])
        train_index = np.array(train_index)
        train_index = set(train_index)
        all_index = [i for i in range(len(gt_1d))]
        all_index = set(all_index)
        background_idx = np.where(gt_1d == 0)[-1]
        background_idx = set(background_idx)
        test_index = all_index - train_index - background_idx
        val_count = int(0.01 * (len(test_index) + len(train_index)))
        val_index = random.sample(test_index, val_count)
        val_index = set(val_index)
        test_index = test_index - val_index
        test_index = list(test_index)
        train_index = list(train_index)
        val_index = list(val_index)

    return train_index, val_index, test_index

class Data_Generator(object):

    def __init__(self, hsi, y, use_coords, batch_size = 24,
                 selection_rules = None, shuffle = True,
                 till_end = False, max_len = 121):
        self.hsi = hsi
        self.y = y
        self.use_coords = use_coords
        self.batch_size = batch_size
        self.selection_rules = selection_rules
        self.shuffle = shuffle
        self.till_end = till_end
        self.max_len = max_len
        self.num_batches = len(self.use_coords) // self.batch_size
        self.on_epoch_end()
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.use_coords))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __len__(self):
        num_data = len(self.use_coords)
        if num_data % self.batch_size == 0:
            return self.num_batches
        elif self.till_end:
            return self.num_batches+1
        else:
            return self.num_batches

    def __getitem__(self, index):
        try:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        except:
            indexes = self.indexes[index*self.batch_size:]

        X_batch, y_batch = self.__data_generation(indexes)
        return X_batch, y_batch

    def __data_generation(self, indexes):
        region = "rect 7"
        coords = self.use_coords[indexes]
        if self.selection_rules is not None:
            region = np.random.choice(self.selection_rules)
        X_batch = Grammar(self.hsi, coords, method=region)
        X_batch_shape = X_batch.shape
        try:
            y_batch = self.y[coords[:, 0], coords[:, 1]]
        except:
            y_batch = self.y[indexes]
        if len(X_batch_shape) == 4:
            X_batch = np.reshape(X_batch, [X_batch_shape[0], X_batch_shape[1] * X_batch_shape[2], X_batch_shape[3]])
        X_batch = zeropad_to_max_len(X_batch, max_len=self.max_len)
        return X_batch, y_batch

def one_hot(gt_mask, height, width):
    gt_one_hot = []
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            temp = np.zeros(2, dtype=np.float32)
            if gt_mask[i, j] != 0:
                temp[int(gt_mask[i, j]) - 1] = 1
            gt_one_hot.append(temp)
    gt_one_hot = np.reshape(gt_one_hot, [height, width, 2])
    return gt_one_hot

def make_mask(gt_mask, height, width):
    label_mask = np.zeros([height * width, 2])
    temp_ones = np.ones([2])
    gt_mask_1d = np.reshape(gt_mask, [height * width])
    for i in range(height * width):
        if gt_mask_1d[i] != 0:
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, 2])
    return label_mask

def get_mask_onehot(gt, index):
    height, width = gt.shape
    gt_1d = np.reshape(gt, [-1])
    gt_mask = np.zeros_like(gt_1d)
    for i in range(len(index)):
        gt_mask[index[i]] = gt_1d[index[i]]
        pass
    gt_mask = np.reshape(gt_mask, [height, width])
    sampling_gt = gt_mask
    gt_onehot = one_hot(gt_mask, height, width)
    gt_mask = make_mask(gt_mask, height, width)
    return gt_onehot, gt_mask, sampling_gt
def get_position(train_coords, height, width, bands):
    position = np.zeros(train_coords.shape[0])
    for i in range(train_coords.shape[0]):
        ps = train_coords[i][0] * width + train_coords[i][1]
        position[i] = ps
    position = position.astype(np.int)
    return position
