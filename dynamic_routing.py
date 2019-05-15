# -*- coding:utf-8 -*-

import tensorflow as tf

def mask_softmax(inputs, length, max_length):
    input_size = inputs.get_shape()[-1].value

    mask = tf.sequence_mask(length, max_length, dtype=tf.float32)  # [batch, max_len]
    mask = tf.expand_dims(mask, axis=1)  # [batch, 1, max_len]
    inputs_temp = tf.nn.softmax(inputs, axis=1)  # [batch, num_caps, max_len]
    inputs_temp = inputs_temp * mask

    return tf.reshape(inputs_temp, [-1, 1, input_size])  # [batch*num_caps, 1, max_len]


class DR(object):
    def __init__(self, vocab_size, embedding_size, max_length,
                 rnn_size, num_capsules, cap_size, num_iter, num_hidden, l2_reg_lambda=0.0):
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)

        self.input_x = tf.placeholder(tf.int32, shape=[None, max_length],
                                 name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, max_length],
                                         name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, [], 'dropout_keep_prob')

        with tf.name_scope("embedding"):
            W_weight = tf.get_variable(name='embedding_weights', shape=[vocab_size, embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
            self.embedding_weight = tf.concat([tf.zeros([1, embedding_size]), W_weight[1:, :]],
                                              axis=0)

            emb  = tf.nn.embedding_lookup(self.embedding_weight, self.input_x, name="embed")
            self.embed = tf.nn.dropout(emb, keep_prob=self.keep_prob)

        with tf.name_scope("sequence_encoder"):
            self.length = self.get_length(self.input_x)

            cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

            output_left, _  = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              self.embed,
                                                              dtype=tf.float32,
                                                              sequence_length=self.length)

            H  = tf.concat(output_left, -1)  # [batch, max_len, 2*rnn_size]

        with tf.name_scope('dr_agg'):  # use dynamic routing for information aggregation
            W_trans = tf.get_variable('W_trans', shape=[2 * rnn_size, cap_size * num_capsules],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      regularizer=regularizer)
            temp = tf.einsum('abc,cd->abd', H, W_trans)
            f = tf.concat(tf.split(temp, num_capsules, axis=-1), axis=0)

            b = tf.zeros(shape=[tf.shape(H)[0], num_capsules, max_length], dtype=tf.float32)

            for i in range(num_iter):
                c = mask_softmax(b, self.length, max_length)  # [batch*num_caps, 1, max_len]
                s = tf.matmul(c, f)  # [batch*num_caps, 1, cap_size]
                s_norm_square = tf.reduce_sum(tf.square(s), keepdims=True)
                v = s / (tf.sqrt(s_norm_square) + 1e-8) *(s_norm_square / (1 + s_norm_square))
                if i == num_iter-1:
                    e_temp = tf.squeeze(v, axis=1)
                    self.e = tf.reshape(e_temp, [-1, cap_size*num_capsules])
                    break

                b = b + tf.reshape(tf.matmul(v, tf.transpose(f, [0, 2, 1])), [-1, num_capsules, max_length])

        with tf.name_scope():
            W = tf.get_variable(
                "W_hidden",
                shape=[cap_size*num_capsules, num_hidden], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
            b = tf.get_variable("b_hidden", shape=[num_hidden], dtype=tf.float32,
                                initializer=tf.zeros_initializer(), regularizer=regularizer)
            output = tf.matmul(self.e, W) + b
            self.full_out = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
            b = tf.get_variable("b_output", shape=[2], dtype=tf.float32,
                                initializer=tf.zeros_initializer(), regularizer=regularizer)
            self.scores = tf.nn.xw_plus_b(self.full_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def get_length(x):
        a = tf.sign(x)
        b = tf.reduce_sum(a, axis=-1)
        return tf.cast(b, dtype=tf.int32)

