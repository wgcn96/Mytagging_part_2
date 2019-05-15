# -*-encoding:utf-8-*-

import tensorflow as tf
import numpy as np
import random
import time
import datetime
import math
import os

from static import *
from Conf import ConfigParser


def saveMatrix(k, matrix):
    fileName = "resultMatrix_k={}".format(k)
    filePath = os.path.join(own_algorithm_dir, fileName)
    np.save(filePath, matrix)


if __name__ == '__main__':

    # global variables

    para_file_path = os.path.join(os.getcwd(), 'settings', '1223server.conf')
    con = ConfigParser(para_file_path)
    res = con.get_config()

    k = int(res['k'])
    batch_size = int(res['batch_size'])
    epoch = int(res['epoch'])
    l2_weight = float(res['l2_weight'])
    learning_rate = float(res['learning_rate'])
    logPath = os.path.join(log_dir, res['logpath'])
    l2_b_extra = float(res['l2_b_extra'])
    l2_extra = float(res['l2_extra'])

    print("Starting Load Rating Matrix")
    print(datetime.datetime.now())

    # 加载四个矩阵
    # rate_matrix  0，1评分矩阵
    # implicit_matrix  隐式反馈矩阵
    # c_matrix  tag之间关联性矩阵
    # theta_matrix  tag之间相似性矩阵

    rate_matrix_path = os.path.join(data_dir, "comprehensive_matrix.npy")
    rate_matrix = np.load(rate_matrix_path).astype(np.float32)
    shape = rate_matrix.shape
    movie_count = shape[0]
    tag_count = shape[1]
    implicit_matrix_path = os.path.join(data_dir, "review_implicit_matrix.npy")
    implicit_matrix = np.load(implicit_matrix_path).astype(np.float32)         # 隐式反馈矩阵

    theta_matrix = np.load(os.path.join(data_dir, "tag_similar_matrix.npy")).astype(np.float32)         # 标签相似性矩阵
    theta_index_matrix = np.load(os.path.join(data_dir, "tag_similar_index_matrix.npy")).astype(np.float32)

    # rate_matrix = np.ones([170, 170], dtype=np.float32)
    # c_matrix = np.random.rand(170, 170).astype(np.float32)
    # implicit_matrix = np.ones([170, 170], dtype=np.float32)
    # theta_matrix = np.ones([170, 170], dtype=np.float32)
    # theta_index_matrix = np.ones([170, 170], dtype=np.float32)

    print("Starting Caculate Average Rating u")

    rate_index_list = []
    rate_list = []
    total = 0
    for i in range(movie_count):
        for j in range(tag_count):
            if rate_matrix[i][j] != 0:
                rate_index_list.append([i, j])
                rate_list.append(rate_matrix[i][j])
                total += rate_matrix[i][j]
    u = float(total) / (movie_count * tag_count)
    x_data = np.array(rate_index_list)      # (N*2)
    y_data = np.array(rate_list)[:, None]   # (N*1)

    print("load matrix finish.")

    u = tf.constant(u, dtype=tf.float32)
    # theta_N = tf.constant(1/10, tf.float32)
    x = tf.placeholder(tf.int32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="Y")
    b_user = tf.Variable(1e-4 * tf.random_uniform([movie_count, 1], -1, 1, dtype=tf.float32), name="b_user")
    b_item = tf.Variable(1e-4 * tf.random_uniform([tag_count, 1], -1, 1, dtype=tf.float32), name="b_item")
    p = tf.Variable(tf.random_uniform([movie_count, k], -1, 1, dtype=tf.float32), name="user_matrix")
    q = tf.Variable(1e-4 * tf.random_uniform([tag_count, k], -1, 1, dtype=tf.float32), name="item_matrix")
    tag_y = tf.Variable(1e-4 * tf.random_uniform([tag_count, k], -1, 1, dtype=tf.float32), name="tag_y")
    c_matrix = tf.Variable(1e-4 * tf.random_uniform([tag_count, tag_count], -1, 1, dtype=tf.float32), name="c_matrix")  # 标签关联矩阵
    # m_ui = tf.convert_to_tensor_or_sparse_tensor
    # m_implicit = tf.convert_to_tensor(implicit_matrix, dtype=tf.float32, name="implicit_matrix")       # 隐式反馈矩阵
    # c = tf.convert_to_tensor(c_matrix, dtype=tf.float32, name="c_matrix")               # 标签关联矩阵

    # rate_matrix_tensor = tf.convert_to_tensor(rate_matrix, dtype=tf.float32)
    # s, p, q = tf.svd(rate_matrix_tensor)
    # p = p[:, :k]
    # q = q[:, :k]

    movie_pos = x[:, 0]
    tag_pos = x[:, 1]
    with tf.device('/cpu:0'):
        target_b_user = tf.nn.embedding_lookup(b_user, movie_pos)
        target_b_item = tf.nn.embedding_lookup(b_item, tag_pos)
        target_p = tf.nn.embedding_lookup(p, movie_pos)
        target_q = tf.nn.embedding_lookup(q, tag_pos)
        target_p = tf.reshape(target_p, [-1, 1, k])
        target_q = tf.reshape(target_q, [-1, k, 1])

    # theta i 部分
    with tf.device('/cpu:0'):
        target_theta_i = tf.nn.embedding_lookup(theta_matrix, tag_pos)      # theta(i)  （？，2015）
        theta_N_rows = tf.nn.embedding_lookup(theta_index_matrix, tag_pos)  # theta_N(i)  （？，2015）

    theta_N = tf.reduce_sum(theta_N_rows, axis=1, keep_dims=True)                       # theta-N(i)  （？，1）
    print(theta_N.shape.as_list())
    # theta_N = tf.sqrt(1/10)
    theta_N = tf.pow(theta_N + 0.01, tf.constant(-0.5))
    # theta_N = tf.multiply(z5, theta_N)
    print(theta_N.shape.as_list())

    target_theta_sum = tf.matmul(target_theta_i, tag_y)                 # （？，2015）* （2015，k）
    target_theta_sum = tf.multiply(theta_N, target_theta_sum)

    # print(target_theta_sum.shape.as_list())                           # result shape （？，k）
    target_q = target_q + tf.reshape(target_theta_sum, [-1, k, 1])      # broadcast
    # print(target_q.shape.as_list())                           # result shape （？，k，1）

    bias = u + target_b_user + target_b_item
    p_e = tf.matmul(target_p, target_q)
    p_e = tf.reshape(p_e, [-1, 1])
    predict = p_e + bias

    # Cij 部分
    with tf.device('/cpu:0'):
        target_u_implicit = tf.nn.embedding_lookup(implicit_matrix, movie_pos)     # Ru 行向量（？，2015）
        target_item_in_c = tf.nn.embedding_lookup(c_matrix, tag_pos)    # Ci 行向量（？，2015）
    target_c_row = tf.multiply(target_u_implicit, target_item_in_c)        # 对应元素乘，0的位置为0，1的位置为Cij。相当于对元素按下标1取值。（？，2015）
    sum_c = tf.reduce_sum(target_c_row, 1, keep_dims=True)                                 # 按行求和  （？，1）
    predict = predict + sum_c

    # 最小化方差
    pre_loss = tf.reduce_sum(tf.square(y - predict))

    # 正则项
    l2_weight = tf.constant(l2_weight)
    l2_target_p = tf.nn.l2_loss(target_p)*2
    l2_target_q = tf.nn.l2_loss(target_q)*2
    l2_target_bu = tf.nn.l2_loss(target_b_user)*2
    l2_target_bi = tf.nn.l2_loss(target_b_item)*2

    # l2_theta_c
    l2_target_sum_c = tf.multiply(target_c_row, target_c_row)
    l2_target_sum_c = tf.reduce_sum(l2_target_sum_c)
    # l2_target_sum_c = tf.sqrt(l2_target_sum_c)
    # l2_target_sum_c = tf.reduce_sum(l2_target_sum_c, 0)

    # l2_theta_y
    with tf.device('/cpu:0'):
        target_theta_index = tf.nn.embedding_lookup(theta_index_matrix, tag_pos)      # theta(i)  （？，2015）
    target_theta_y = tf.matmul(target_theta_index, tag_y)                           # （？，k）
    l2_target_theta_y_sum = tf.multiply(target_theta_y, target_theta_y)
    l2_target_theta_y_sum = tf.reduce_sum(l2_target_theta_y_sum)
    # l2_target_theta_y_sum = tf.sqrt(l2_target_theta_y_sum)
    # l2_target_theta_y_sum = tf.reduce_sum(l2_target_theta_y_sum, 0)

    l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q + l2_b_extra * l2_weight*l2_target_bu + \
             l2_b_extra * l2_weight*l2_target_bi + l2_extra * l2_weight*l2_target_sum_c + l2_extra * l2_weight*l2_target_theta_y_sum

    loss = pre_loss + l2_all
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # tensorboard
    # 记录loss曲线
    tf.summary.scalar('pre_loss', pre_loss)
    tf.summary.scalar('total_loss', loss)
    tf.summary.histogram('b_user', b_user)
    tf.summary.histogram('b_item', b_item)
    tf.summary.histogram('p_user', p)
    tf.summary.histogram('q_item', q)
    tf.summary.histogram('c_matrix', c_matrix)
    tf.summary.histogram('tag_y', tag_y)

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
        total_step = epoch*len(rate_list)
        print("total step : {}".format(total_step))

        for step in range(total_step):

            index = permutation[step % len(rate_list)]

            sess.run(train, feed_dict={x: (x_data[index]).reshape(-1, 2), y: (y_data[index]).reshape(-1, 1)})

            if step % 10000 == 1:
                print("current step : {}".format(step))
                predict_loss = sess.run(pre_loss, feed_dict={x: x_data, y: y_data})
                # total_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})

                # 防止过拟合
                current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
                if total_loss - current_loss < 10:
                    count += 1
                    print(" ***")
                    if count >= 10:
                        break
                else:
                    count = 0
                total_loss = current_loss

                # pre_loss：预测loss
                # total_loss = loss + l2_loss
                print("pre_loss ", predict_loss)
                print("total_loss ", total_loss)

                rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
                writer.add_summary(rs, step)
                pass        # end id
            pass    # end for

        # 生成结果矩阵
        result_matrix = np.zeros((movie_count, tag_count), dtype=np.float32)
        for i in range(movie_count):
            rate_index_list = []
            for j in range(tag_count):
                rate_index_list.append([i, j])

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

        saveMatrix(k, result_matrix)
        np.savetxt(os.path.join(logPath, "b_user"), b_user.eval(session=sess))
        np.savetxt(os.path.join(logPath, "b_item"), b_item.eval(session=sess))
        np.savetxt(os.path.join(logPath, "p"), p.eval(session=sess))
        np.savetxt(os.path.join(logPath, "q"), q.eval(session=sess))
        np.savetxt(os.path.join(logPath, "c_matrix"), c_matrix.eval(session=sess))
        np.savetxt(os.path.join(logPath, "tag_y"), tag_y.eval(session=sess))

    print(datetime.datetime.now())
    print("finish!")


