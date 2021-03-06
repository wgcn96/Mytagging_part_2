# -*-encoding:utf-8-*-
"""
计算p*q+b
_author: wang chen
"""

import tensorflow as tf
import numpy as np
import time
import datetime
import math
import os


from SVD_3.static import *
from script.Conf import ConfigParser


def saveMatrix(k, matrix, fileName = None):
    if fileName is None:
        fileName = "resultMatrix_k={}".format(k)
    filePath = os.path.join(data_dir, fileName)
    np.save(filePath, matrix)


def getSample(array, n):
    result = []
    for i in range(n):
        index = np.random.randint(len(array))
        result.append(index)
    return result


def fillDir(basedir, filename, n):
    full_path = os.path.join(basedir, str(n), filename)
    print(full_path)
    return full_path


if __name__ == '__main__':

    # global variables

    para_file_path = os.path.join(os.getcwd(), 'settings', '0323.conf')
    con = ConfigParser(para_file_path)
    res = con.get_config()

    k = int(res['k'])
    batch_size = int(res['batch_size'])
    epoch = float(res['epoch'])
    l2_weight = float(res['l2_weight'])
    learning_rate = float(res['learning_rate'])
    l2_b_extra = float(res['l2_b_extra'])
    sample_n = int(res['sample_n'])
    logPath = fillDir(log_dir, res['logpath'], sample_n)

    # rate_matrix = np.array([[1,0,1],[0,1,0],[0,1,1]])

    print("Starting Load Rating Matrix")
    print(datetime.datetime.now())

    # rate_matrix = np.load(fillDir(data_dir, 'comprehensive_index_matrix_sample.npy', sample_n))
    rate_matrix = np.load(os.path.join(data_dir, 'com_matrix_sample_top250_ten.npy'))

    print("matrix shape", rate_matrix.shape)
    # rate_matrix = np.ones([170,170], dtype=np.int32)
    print("Starting Caculate Average Rating u")
    movie_count = rate_matrix.shape[0]
    tag_count = rate_matrix.shape[1]
    rate_index_list = []
    neg_rate_index_list = []
    rate_list = []
    neg_rate_list = []
    total = 0
    for i in range(movie_count):
        for j in range(tag_count):
            if rate_matrix[i][j] != -1:
                if rate_matrix[i][j] > 0:
                    rate_index_list.append([i, j])
                    rate_list.append(rate_matrix[i][j])
                    total += rate_matrix[i][j]
                else:
                    neg_rate_index_list.append([i, j])
                    neg_rate_list.append(rate_matrix[i][j])

    u = float(total) / (movie_count * tag_count)
    x_data = np.array(rate_index_list)      # (N*2)
    y_data = np.array(rate_list)[:, None]   # (N*1)
    neg_x_data = np.array(neg_rate_index_list)
    neg_y_data = np.array(neg_rate_list)[:, None]
    print("load matrix finish.")

    total_step = int(epoch * len(rate_list))

    u = tf.constant(u, dtype=tf.float32)
    x = tf.placeholder(tf.int32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="Y")
    # b_user = tf.Variable(1e-3 * tf.random_uniform([movie_count, 1], 0, 1, dtype=tf.float32), name="b_user")
    # b_item = tf.Variable(1e-3 * tf.random_uniform([tag_count, 1], 0, 1, dtype=tf.float32), name="b_item")
    # p = tf.Variable(1e-3 * tf.random_uniform([movie_count, k], 0, 1, dtype=tf.float32), name="user_matrix")
    # q = tf.Variable(1e-3 * tf.random_uniform([tag_count, k], 0, 1, dtype=tf.float32), name="item_matrix")

    b_user = tf.Variable(tf.random_uniform([movie_count, 1], -1e-2, 1e-2, dtype=tf.float32), name="b_user")
    b_item = tf.Variable(tf.random_uniform([tag_count, 1], -1e-2, 1e-2, dtype=tf.float32), name="b_item")
    # p = tf.Variable(tf.random_uniform([movie_count, k], -0.5, 0.5, dtype=tf.float32), name="user_matrix")
    # q = tf.Variable(tf.random_uniform([tag_count, k], -0.5, 0.5, dtype=tf.float32), name="item_matrix")
    p = tf.Variable(np.load(os.path.join(log_dir, 'svd_1229_3', 'p.npy')), name="user_matrix")
    q = tf.Variable(np.load(os.path.join(log_dir, 'svd_1229_3', 'q.npy')), name="item_matrix")

    with tf.name_scope('pre') as scope:
        movie_pos = x[:, 0]
        tag_pos = x[:, 1]
        target_b_user = tf.nn.embedding_lookup(b_user, movie_pos)
        target_b_item = tf.nn.embedding_lookup(b_item, tag_pos)
        target_p = tf.nn.embedding_lookup(p, movie_pos)
        target_q = tf.nn.embedding_lookup(q, tag_pos)
        target_p = tf.reshape(target_p, [-1,1,k])
        target_q = tf.reshape(target_q, [-1,k,1])
        bias = u + target_b_user + target_b_item
        p_e = tf.matmul(target_p, target_q)
        p_e = tf.reshape(p_e, [-1, 1])
        predict = p_e + bias
        # predict = p_e

    # 最小化方差
    with tf.name_scope('loss') as scope:
        pre_loss = tf.reduce_sum(tf.square(y - predict))

        # 正则项
        l2_weight = tf.constant(l2_weight)
        l2_target_p = tf.nn.l2_loss(target_p)*2
        l2_target_q = tf.nn.l2_loss(target_q)*2
        l2_target_bu = tf.nn.l2_loss(target_b_user)*2
        l2_target_bi = tf.nn.l2_loss(target_b_item)*2
        l2_all = l2_weight*l2_target_p + 10 * l2_weight*l2_target_q + l2_b_extra * l2_weight * l2_target_bu + 10 * l2_b_extra * l2_weight * l2_target_bi
        # l2_all = l2_weight * l2_target_p + 10 * l2_weight * l2_target_q

        loss = pre_loss + l2_all

    current_step = tf.Variable(0, trainable=True)
    learning_rate = tf.train.exponential_decay(learning_rate, current_step, 20000, 0.96, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=current_step)
    neg_train = optimizer.minimize(loss)

    # tensorboard
    tf.summary.scalar('pre_loss', pre_loss)
    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.histogram('b_user', b_user)
    tf.summary.histogram('b_item', b_item)
    tf.summary.histogram('p_user', p)
    tf.summary.histogram('q_item', q)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 启动图 (graph)
    sess = tf.Session()
    with sess.as_default():
        writer = tf.summary.FileWriter(logPath, sess.graph)

        sess.run(init)
        merged = tf.summary.merge_all()

        permutation = np.random.permutation(len(x_data))
        total_loss = 100000000000
        count = 0

        current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
        print(current_loss)

        print("total step : {}".format(total_step))
        for step in range(total_step):
            # current_step = step
            index = permutation[step % len(rate_list)]

            sess.run(train, feed_dict={x: (x_data[index]).reshape(-1, 2), y: (y_data[index]).reshape(-1, 1)})

            neg_index = getSample(neg_rate_list, n=4)
            sess.run(neg_train, feed_dict={x: (neg_x_data[neg_index]).reshape(-1, 2), y: (neg_y_data[neg_index]).reshape(-1, 1)})

            if step % 100000 == 0:
                # current_step = step
                print("current step : {}".format(step))
                predict_loss = sess.run(pre_loss, feed_dict={x: x_data, y: y_data})
                total_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})

                # 防止过拟合
                # current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
                # if total_loss - current_loss < 10:
                #     count += 1
                #     print(" ***")
                #     if count >= 10:
                #         break
                # else:
                #     count = 0
                # total_loss = current_loss

                # pre_loss：预测loss
                # total_loss = loss + l2_loss
                print("pre_loss ", predict_loss)
                print("total_loss ", total_loss)

                rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
                writer.add_summary(rs, step)
                pass        # end id
            pass    # end for

        result_matrix = np.zeros((movie_count, tag_count), dtype=np.float32)
        for i in range(movie_count):
            rate_index_list = []
            for j in range(tag_count):
                rate_index_list.append([i, j])
                # print(len(rate_index_list))
            x_pre = np.array(rate_index_list)
            cur_row = sess.run(predict, feed_dict={x: x_pre})
            result_matrix[i] = cur_row[:, 0]
        result_matrix = result_matrix.reshape(movie_count, tag_count)

        result_matrix_wt_bias = np.zeros((movie_count, tag_count), dtype=np.float32)
        for i in range(movie_count):
            rate_index_list = []
            for j in range(tag_count):
                rate_index_list.append([i, j])
                # print(len(rate_index_list))
            x_pre = np.array(rate_index_list)
            cur_row = sess.run(p_e, feed_dict={x: x_pre})
            result_matrix_wt_bias[i] = cur_row[:, 0]
        result_matrix_wt_bias = result_matrix_wt_bias.reshape(movie_count, tag_count)

        index_matrix = np.argsort(-result_matrix, axis=1)
        index_matrix_wt_bias = np.argsort(-result_matrix_wt_bias, axis=1)

        fileName = "resultMatrix_k={}_sample_{}".format(k, sample_n)
        saveMatrix(k, result_matrix, fileName)
        value_p = p.eval(session=sess)
        value_q = q.eval(session=sess)
        value_bu = b_user.eval(session=sess)
        value_bi = b_item.eval(session=sess)
        np.savetxt(os.path.join(logPath, "b_user"), value_bu)
        np.savetxt(os.path.join(logPath, "b_item"), value_bi)
        np.savetxt(os.path.join(logPath, "p"), value_p)
        np.savetxt(os.path.join(logPath, "q"), value_q)

    print(datetime.datetime.now())
    print("finish!")

