
# coding: utf-8

import tensorflow as tf
import math
import data_helpers
import sequence_labelling_model_bidirectional

sequences, labels = data_helpers.load_and_pad_seqences_and_labels()
sequences, labels = data_helpers.shuffle_data_and_labels(sequences, labels)
seq_train, seq_dev, labels_train, labels_dev = data_helpers.partition_data_and_labels(sequences, labels, 0.1)

x = tf.placeholder(dtype=tf.float32, shape=[None] + list(sequences.shape)[1:])
y = tf.placeholder(dtype=tf.float32, shape=[None] + list(labels.shape)[1:])

def GRU_cell(cell_size):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
    return rnn_cell    

params = dict()
params["rnn_cell"] = GRU_cell
params["rnn_cell_size"] = 200
params["optimizer"] = tf.train.RMSPropOptimizer(0.002)
params["gradient_clipping"] = 0.0

model = sequence_labelling_model_bidirectional.SequenceLabellingModel(x, 
                                                                      y, 
                                                                      params)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    num_epochs = 100
    for epoch in range(num_epochs):
        print("-epoch {}:".format(epoch))
        generator = data_helpers.batch_generator(seq_train, 
                                                 labels_train, 
                                                 100)
        for i, seq_batch, labels_batch in generator:
            feed_dict = {}
            feed_dict[model.sequences] = seq_batch
            feed_dict[model.labels] = labels_batch
            _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy], 
                                    feed_dict)
#             print("{} - loss: {}, accuracy: {}".format(i, loss, acc))
        print("evaluating...")
        feed_dict = {}
        feed_dict[model.sequences] = seq_dev
        feed_dict[model.labels] = labels_dev
        loss, acc = sess.run([model.loss, model.accuracy], 
                                    feed_dict)
        print("loss: {}, accuracy: {}".format(loss, acc))