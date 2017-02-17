"""
Copyright (c) 2017 Claudio Pomo

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def uniformize(samples, labels):
    """ Getting the minimum counter foreach samples,
    in training dataset, and truncate all other
    to this threshold
    """
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1

    minimun = count[min(count, key=count.get)]

    for i in range(1, 11):
        indeces = np.where(labels == [i])
        if (len(indeces[0]) - minimun) > 0:
            indeces_final = np.random.choice(indeces[0], (len(indeces[0]) - minimun), replace=False)
            samples = np.delete(samples, indeces_final, 3)
            labels = np.delete(labels, indeces_final, 0)

    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1

    return samples, labels


def reformat(samples, labels):
    """ Reshape samples in all dataset in this way (0,1,2,3) -> (3,0,1,2)
    and encode labels set in one-hot set taking care to translate
    label 10 to 0
    """
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
    """ Reduce samples dimension in all dataset from this shape (size,32,32,3) to
    (size,32,32,1)
    """
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a / 3.0
    return a / 128.0 - 1.0


def distribution(labels, semples, name):
    """ Evaluate distribution of each keys: 0, 1, ..., 9 in each
    dataset, and print mean and standard deviation for this
    """
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1

    print('------------',name,'--------------')
    for i in range(1, 11):
        indeces = np.where(labels == [i])
        semple = semples[indeces[0]]
        print('----------------------------------')
        print('Label:', i, 'number sample: ', len(semple))
        print('Mean:', np.mean(semple))
        print('Standard deviation:', np.std(semple))
        print('----------------------------------')

    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    fig = plt.figure()
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()
    fig.savefig(name, dpi=fig.dpi)


def inspect(dataset, labels, i):
    """ Show a resampled image from index i"""
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels)
    plt.imshow(dataset[i])
    plt.show()


train = load('/home/claudio/Documents/DiLecce/svhn/train_32x32.mat')
test = load('/home/claudio/Documents/DiLecce/svhn/test_32x32.mat')
extra = load('/home/claudio/Documents/DiLecce/svhn/extra_32x32.mat')

print('Train Samples Shape:', train['X'].shape)
print('Train Labels Shape:', train['y'].shape)

print('Test Samples Shape:', test['X'].shape)
print('Test Labels Shape:', test['y'].shape)

print('Extra Train Samples Shape:', extra['X'].shape)
print('Extra Train Labels Shape:', extra['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
extra_samples = extra['X']
extra_labels = extra['y']

train_samples = np.concatenate((train_samples, extra_samples), axis=3)
train_labels = np.concatenate((train_labels, extra_labels))

train_samples, train_labels = uniformize(train_samples, train_labels)

n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    pass
    # _train_samples = normalize(_train_samples)
    inspect(_train_samples, _train_labels, 1234)
    distribution(train_labels, _train_samples, 'Train Labels')
    distribution(test_labels, _test_samples, 'Test Labels')
