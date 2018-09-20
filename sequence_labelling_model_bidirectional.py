
# coding: utf-8


import tensorflow as tf


class SequenceLabellingModel:
    def __init__(self, sequences, labels, params):
        self.sequences = sequences
        self.labels = labels
        self.params = params
        self.lengths = self.lengths()
        self.scores = self.scores()
        self.loss = self.loss()
        self.accuracy = self.accuracy()
        self.optimizer = self.optimizer()
       
    # Computes actual lengths of sequences.
    def lengths(self):
        binarized = tf.sign(tf.reduce_max(tf.abs(self.sequences), axis=2))
        lengths = tf.reduce_sum(binarized, axis=1)
        lengths = tf.cast(lengths, tf.int32)
        return lengths
    
    def scores(self):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.params["rnn_cell"](self.params["rnn_cell_size"]), 
                                                     self.params["rnn_cell"](self.params["rnn_cell_size"]),
                                                     self.sequences, 
                                                     dtype=tf.float32, 
                                                     sequence_length=self.lengths)
        outputs = tf.concat(outputs, 2)
        # Flatten the outputs so that a shared softmax can be applied to each element in the sequence.
        outputs = tf.reshape(outputs, [-1, self.params["rnn_cell_size"] * 2])
        num_classes = int(self.labels.shape[2])
        max_length = int(self.labels.shape[1])
        Weights = tf.Variable(tf.truncated_normal([self.params["rnn_cell_size"] * 2, num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        scores = tf.nn.xw_plus_b(outputs, Weights, bias)
        scores = tf.reshape(scores, [-1, max_length, num_classes])
        return scores 
    
    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, 
                                                                   labels=self.labels)
        # Mask off the padded parts.
        mask = tf.reduce_max(self.labels, axis=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        cross_entropy /= tf.cast(self.lengths, tf.float32)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy
    
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.scores, axis=2), 
                                       tf.argmax(self.labels, axis=2))
        correct_predictions = tf.cast(correct_predictions, tf.float32)
        mask = tf.reduce_max(self.labels, axis=2)
        correct_predictions *= mask
        correct_predictions = tf.reduce_sum(correct_predictions, axis=1)
        correct_predictions /= tf.cast(self.lengths, tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)
        return accuracy
    
    def optimizer(self):
        gradients = self.params["optimizer"].compute_gradients(self.loss)
        if self.params["gradient_clipping"]:
            limit = self.params["gradient_clipping"]
            gradients = [(tf.clip_by_value(grad, -limit, limit), var)                          if grad                          else (None, var)                          for gard, var in gradients]
        optimizer = self.params["optimizer"].apply_gradients(gradients)
        return optimizer 