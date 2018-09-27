from copy import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))

from utils.evaluator import evaluate
from model.base_predictor import BasePredictor

class LSTM(BasePredictor):
    def __init__(self, parameters):
        BasePredictor.__init__(self, parameters=parameters)
        self.gpu = (self.pars.gpu == 1)
        self.reload = (self.pars.reload == 1)
        self.fix_init = (self.pars.fix_init == 1)

    def _get_batch(self, X_tr, y_tr, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, X_tr.shape[0])
        if sta_ind + self.pars.batch_size < y_tr.shape[0]:
            end_ind = sta_ind + self.pars.batch_size
        else:
            sta_ind = X_tr.shape[0] - self.pars.batch_size
            end_ind = y_tr.shape[0]
        return X_tr[sta_ind:end_ind, :, :], y_tr[sta_ind:end_ind, :]

    def construct_graph(self):
        if self.pars.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.pars.fix_init:
                tf.set_random_seed(0)

            self.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.placeholder(
                tf.float32, [None, self.pars.lag, self.fea_dim]
            )

            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.pars.unit
            )

            # self.in_lat = tf.layers.dense(
            #     self.pv_var, units=self.fea_dim,
            #     activation=tf.nn.tanh, name='in_fc',
            #     kernel_initializer=tf.glorot_uniform_initializer()
            # )
            #
            # self.outputs, _ = tf.nn.dynamic_rnn(
            #     # self.outputs, _ = tf.nn.static_rnn(
            #     self.lstm_cell, self.in_lat, dtype=tf.float32
            #     # , initial_state=ini_sta
            # )

            self.outputs, _ = tf.nn.dynamic_rnn(
                # self.outputs, _ = tf.nn.static_rnn(
                self.lstm_cell, self.pv_var, dtype=tf.float32
                # , initial_state=ini_sta
            )

            self.pred = tf.layers.dense(
                self.outputs[:, -1, :], units=1, activation=tf.nn.sigmoid,
                name='pre_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            # self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
            self.loss = tf.losses.log_loss(self.gt_var, self.pred)

            self.l2_norm = 0
            self.tra_vars = tf.trainable_variables('pre_fc')
            for var in self.tra_vars:
                self.l2_norm += tf.nn.l2_loss(var)

            self.obj_func = self.loss + \
                            self.pars.alpha_l2 * self.l2_norm

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.pars.learning_rate
            ).minimize(self.obj_func)

    def test(self, X_tes, y_tes):
        self.fea_dim = X_tes.shape[2]
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.pars.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # # test on validation set
        # feed_dict = {
        #     self.pv_var: self.val_pv,
        #     self.gt_var: self.val_gt
        # }
        # val_loss, val_pre = sess.run(
        #     (self.loss, self.pred), feed_dict
        # )
        # cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        # print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

        # test on testing set
        feed_dict = {
            self.pv_var: X_tes,
            self.gt_var: y_tes
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, y_tes)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        sess.close()
        tf.reset_default_graph()

    def train(self, X_tr, y_tr, X_val, y_val, tune_para=False):
        self.fea_dim = X_tr.shape[2]
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.pars.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(X_val.shape, dtype=float)
        # best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        # best_test_perf = {
        #     'acc': 0, 'mcc': -2
        # }

        bat_count = X_tr.shape[0] // self.pars.batch_size
        if not (X_tr.shape[0] % self.pars.batch_size == 0):
            bat_count += 1
        for i in range(self.pars.epoch):
            t1 = time()
            # first_batch = True
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_acc = 0.0
            for j in range(bat_count):
                pv_b, gt_b = self._get_batch(
                    X_tr, y_tr, j * self.pars.batch_size
                )
                feed_dict = {
                    self.pv_var: pv_b,
                    self.gt_var: gt_b
                }
                cur_pre, cur_obj, cur_loss, cur_l2, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm,
                     self.optimizer),
                    feed_dict
                )
                cur_tra_perf = evaluate(cur_pre, gt_b)
                tra_acc += cur_tra_perf['acc']
                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
            print('----->>>>> Training:', tra_obj / bat_count,
                  tra_loss / bat_count, l2 / bat_count, '\tTrain per:',
                  tra_acc / bat_count)

            # if not tune_para:
            #     tra_loss = 0.0
            #     tra_obj = 0.0
            #     l2 = 0.0
            #     tra_acc = 0.0
            #     for j in range(bat_count):
            #         pv_b, wd_b, gt_b = self.get_batch(
            #             j * self.batch_size)
            #         feed_dict = {
            #             self.pv_var: pv_b,
            #             self.wd_var: wd_b,
            #             self.gt_var: gt_b
            #         }
            #         cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
            #             (self.obj_func, self.loss, self.l2_norm, self.pred),
            #             feed_dict
            #         )
            #         cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
            #         tra_loss += cur_loss
            #         l2 += cur_l2
            #         tra_obj += cur_obj
            #         tra_acc += cur_tra_perf['acc']
            #         # print('\t\t', cur_loss)
            #     print('Training:', tra_obj / bat_count, tra_loss / bat_count,
            #           l2 / bat_count, '\tTrain per:', tra_acc / bat_count)

            # test on validation set
            feed_dict = {
                self.pv_var: X_val,
                self.gt_var: y_val
            }
            val_loss, val_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
            cur_valid_perf = evaluate(val_pre, y_val)
            print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

            # # test on testing set
            # feed_dict = {
            #     self.pv_var: self.tes_pv,
            #     self.wd_var: self.tes_wd,
            #     self.gt_var: self.tes_gt
            # }
            # test_loss, tes_pre = sess.run(
            #     (self.loss, self.pred), feed_dict
            # )
            # cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
            # print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)

            if cur_valid_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy(cur_valid_perf)
                best_valid_pred = copy(val_pre)
                # best_test_perf = copy.copy(cur_test_perf)
                # best_test_pred = copy.copy(tes_pre)
                if not tune_para:
                    saver.save(sess, self.pars.model_save_path)
            X_tr, y_tr = shuffle(
                X_tr, y_tr, random_state=0
            )
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        print('\nBest Valid performance:', best_valid_perf)
        # print('\tBest Test performance:', best_test_perf)

        # saver.save(sess, self.model_save_path)

        sess.close()
        tf.reset_default_graph()
        if tune_para:
            return best_valid_perf #, best_test_perf
        return best_valid_pred #, best_test_pred

    def update_model(self, parameters):
        data_update = False
        for name, value in self.pars.items():
            self.pars[name] = value
        return True