from copy import copy
import numpy as np
import os
import sys
import json
import random
from data.load_rnn import load_pure_lstm
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.read_data import read_single_csv, merge_data_frame, \
    process_missing_value_v3
from utils.normalize_feature import log_1d_return, normalize_volume, normalize_3mspot_spread, \
    normalize_OI, normalize_3mspot_spread_ex
from utils.transform_data import flatten
from utils.construct_data import construct
import tensorflow as tf

# Hyper Parameters
learning_rate = 0.001    # 学习率
n_steps = 30            # LSTM 展开步数（时序持续长度）
n_inputs = 5           # 输入节点数
n_hiddens = 20         # 隐层节点数
n_layers = 2            # LSTM layer 层数
n_classes = 2          # 输出节点数（分类数目）

tra_date ='2007-01-03'
val_date = '2015-01-02'
tes_date = '2016-01-04'
split_dates = [tra_date, val_date, tes_date]
	# read data configure file
with open("D:/Internship/NExT/4EBaseMetal/exp/lstm_data.conf") as fin:
	fname_columns = json.load(fin)
print(fname_columns)
for fname in fname_columns:
	print('read columns:', fname_columns[fname], 'from:', fname)
		# time_series = read_single_csv(fname, fname_columns[fname])
		# print(time_series)
	# load data
X_tr, y_tr, X_val, y_val, X_tes, y_tes = load_pure_lstm(fname_columns, 'LMCADY', 'log_1d_return', split_dates, n_steps, 1)
	#print(X_tr.shape[2])

	# tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps , n_inputs], name='x_input')     # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')               # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')       # 批大小

# weights and biases
with tf.name_scope('weights'):
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes],stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)
with tf.name_scope('biases'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)
def get_batch( X_tr, y_tr, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, X_tr.shape[0])
        if sta_ind + 512 < y_tr.shape[0]:
            end_ind = sta_ind + 512
        else:
            sta_ind = X_tr.shape[0] - 512
            end_ind = y_tr.shape[0]
        return X_tr[sta_ind:end_ind, :, :], y_tr[sta_ind:end_ind, :]
# RNN structure
def RNN_LSTM(x, Weights, biases):
    # RNN 输入 reshape
    #x = tf.reshape(x, [-1, n_steps, n_inputs])
    # 定义 LSTM cell
    # cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # 实现多层 LSTM
    # [attn_cell() for _ in range(n_layers)]
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    # 输出
    #return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.nn.softmax(tf.matmul(outputs[:,-1,:], Weights) + biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)
# cost
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("logs",sess.graph)
    test_writer = tf.summary.FileWriter("log_test",sess.graph)
    # training
    step = 1
    for i in range(500):
        _batch_size = 512
        batch_x, batch_y = get_batch(X_tr, y_tr)

        sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5, batch_size:_batch_size})
        loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
        acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
        print('Iter: %d' % i, '| train loss: %.6f' % loss, '| train accuracy: %.6f' % acc)
        train_result = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
        test_result = sess.run(merged, feed_dict={x:X_tes, y:y_tes, keep_prob:1.0, batch_size:X_tes.shape[0]})
        train_writer.add_summary(train_result,i+1)
        test_writer.add_summary(test_result,i+1)

    print("Optimization Finished!")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_tes, y:y_tes, keep_prob:1.0, batch_size:X_tes.shape[0]}))
   