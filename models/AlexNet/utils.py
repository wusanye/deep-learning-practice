###########################################
# by Wu Kai, 2018-07-30
##########################################

import os
import numpy as np
import _pickle as pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_dataset(dataset_dir, class_num):
    if not os.path.isdir(dataset_dir):
        print("Error: input data set directory is invalid!")

    train_batches = [os.path.join(dataset_dir, "data_batch_"+str(i)) for i in range(1, 6)]

    xlist, ylist = [], []
    for batch in train_batches:
        d = unpickle(batch)
        xlist.append(d[b"data"])
        ylist.append(d[b"labels"])

    x_train = np.vstack(xlist)
    y_train = np.vstack(ylist)

    with open(os.path.join(dataset_dir, "test_batch"), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        x_test, y_test = d[b"data"], d[b"labels"]

    y_train = np.reshape(y_train, (-1))
    y_test = np.array(y_test).reshape(-1)

    y_train = np.eye(class_num)[y_train]
    y_test = np.eye(class_num)[y_test]

    x_test = x_test[0::2, :]
    y_test = y_test[0::2, :]

    x_val = x_test[1::2, :]
    y_val = y_test[1::2, :]

    return x_train, y_train, x_test, y_test, x_val, y_val


def preprocess_data(data, labels, size):
    height, width, depth = size[0], size[1], size[2]

    mu = np.mean(data, axis=0)
    mu = mu.reshape(1, -1)

    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)

    data = data - mu
    data = data / sigma

    data = data.reshape(-1, depth, height, width)
    data = data.transpose([0, 2, 3, 1])

    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    return data, labels


def get_batch(data, labels, batch_size, batch_no):
    return data[(batch_size*batch_no): (batch_size*(batch_no+1)), :, :, :], \
           labels[(batch_size*batch_no): (batch_size*(batch_no+1)), :]



