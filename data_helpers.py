
# coding: utf-8

import gzip
import csv
import numpy as np
import math

def load_and_pad_seqences_and_labels():
    gzip_file_dir = "./dataset/letter.data.gz"
    with gzip.open(gzip_file_dir, "rt") as file_:
        reader = csv.reader(file_, delimiter="\t")
        lines = list(reader)
    lines = sorted(lines, key=lambda x : int(x[0]))
    sequences = []
    labels = []
    next_ = None
    for line in lines:
        if next_ == None:
            sequences.append([])
            labels.append([])
        else:
            assert next_ == int(line[0]), "data sequence pointer error"
        next_ = int(line[2]) if int(line[2]) > -1 else None
        pixels = np.array(list(map(int, line[6:134])))
#         pixels = pixels.reshape([16, 8])
        sequences[-1].append(pixels)
        labels[-1].append(line[1])

    # pad 
    max_len = max(len(x) for x in labels)
    padding = np.zeros([128])
    sequences = [x + ([padding] * (max_len - len(x))) for x in sequences]
    labels = [x + ([""] * (max_len - len(x))) for x in labels]
    sequences = np.array(sequences)
    labels = np.array(labels)
    # one-hotize labels
    one_hotized_labels = np.zeros(labels.shape + (26,))
    for index, letter in np.ndenumerate(labels):
        if letter:
            one_hotized_labels[index][ord(letter) - ord("a")] = 1
    labels = one_hotized_labels
    return sequences, labels

# partition data into training set and development set according to dev_sample_percentage
def partition_data_and_labels(data, labels, dev_sample_percentage):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    dev_sample_start = int(float(len(data)) * (1 - dev_sample_percentage))
    return data[:dev_sample_start], data[dev_sample_start:], labels[:dev_sample_start], labels[dev_sample_start:]

# shuffles a data labels pair, returns shuffled np.arrays
def shuffle_data_and_labels(data, labels):
    assert len(data) == len(labels), "shuffle: length of data doesn't equal length of labels"
    data = np.array(data)
    labels = np.array(labels)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    return data[shuffle_indices], labels[shuffle_indices]

# batch generator
def batch_generator(data, labels, batch_size):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    num_of_batches = math.ceil(float(len(data)) / float(batch_size))
    for i in range(num_of_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(data))
        yield i, data[start_index : end_index], labels[start_index : end_index]