
from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    #  0       1       2      3          3       0       1      2
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

    # labels one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 , represented as 10
    # labels one-hot encoding, [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])  # slow code, whatever
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels


def normalize(samples):
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a / 3.0
    return a / 128.0 - 1.0


def distribution(labels, semples, name):
    # keys:
    # 0
    # 1
    # 2
    # ...
    # 9
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    for i in range(1, 11):
        indeces = np.where(labels == [i])
        semple = semples[indeces[0]]
        print('----------------------------------')
        print('Label:', i)
        print('Mean:', np.mean(semple))
        print('Standard deviation:', np.std(semple))
        # inspect(semples, labels, indeces[0][6])
        print('----------------------------------')
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels)
    plt.imshow(dataset[i])
    plt.show()


train = load('/home/claudio/Documents/DiLecce/svhn/train_32x32.mat')
test = load('/home/claudio/Documents/DiLecce/svhn/test_32x32.mat')
valid = load('/home/claudio/Documents/DiLecce/svhn/extra_32x32.mat')
# extra = load('../data/extra_32x32.mat')

# print('Train Samples Shape:', train['X'].shape)
# print('Train  Labels Shape:', train['y'].shape)

# print('Train Samples Shape:', test['X'].shape)
# print('Train  Labels Shape:', test['y'].shape)

# print('Train Samples Shape:', extra['X'].shape)
# print('Train  Labels Shape:', extra['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
valid_samples = valid['X']
valid_labels = valid['y']
# extra_samples = extra['X']
# extra_labels = extra['y']

n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)
n_valid_samples, _valid_labels = reformat(valid_samples, valid_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)
_valid_samples = normalize(n_valid_samples)

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    pass
    # _train_samples = normalize(_train_samples)
    # inspect(_train_samples, _train_labels, 1234)
    distribution(train_labels, _train_samples, 'Train Labels')
    distribution(test_labels, _test_samples, 'Test Labels')
    distribution(_valid_labels, _valid_samples, 'Test Labels')
