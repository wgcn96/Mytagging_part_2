# -*-encoding:utf-8-*-

import tensorflow as tf
import numpy as np
import random
import time
import datetime
import math
import os


from static import *

def saveMatrix(k, matrix, fileName = None):
    if fileName is None:
        fileName = "resultMatrix_k={}".format(k)
    filePath = os.path.join(data_dir, fileName)
    np.save(filePath, matrix)


if __name__ == '__main__':

    # global variables
    k = 10
    batch_size = 1000
    epoch = 100
    l2_weight = 1e-3
    learning_rate = 0.005
    logPath = os.path.join(log_dir, "svd_k10_1213")

    # rate_matrix = np.array([[1,0,1],[0,1,0],[0,1,1]])

    print("Starting Load Rating Matrix")
    print(datetime.datetime.now())
    rate_matrix = np.load(matrix_path)
    # rate_matrix = np.ones([170,170], dtype=np.int32)
    print("Starting Caculate Average Rating u")
    movie_count = rate_matrix.shape[0]
    tag_count = rate_matrix.shape[1]
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
    x = tf.placeholder(tf.int32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="Y")
    b_user = tf.Variable(1e-2 * tf.random.uniform([movie_count, 1], 0, 1, dtype=tf.float32), name="b_user")
    b_item = tf.Variable(1e-2 * tf.random.uniform([tag_count, 1], 0, 1, dtype=tf.float32), name="b_item")
    # p = tf.Variable(tf.random.uniform([movie_count, k], 0, 1, dtype=tf.float32), name="user_matrix")
    # q = tf.Variable(tf.random.uniform([tag_count, k], 0, 1, dtype=tf.float32), name="item_matrix")

    rate_matrix_tensor = tf.convert_to_tensor(rate_matrix, dtype=tf.float32)
    s, p, q = tf.svd(rate_matrix_tensor)
    p = p[:, :k]
    q = q[:, :k]

    '''
    target_b_user = tf.nn.embedding_lookup(b_user, x[:, 0:1])
    target_b_item = tf.nn.embedding_lookup(b_item, x[:, 1:2])
    target_p = tf.nn.embedding_lookup(p, x[:, 0:1])
    target_q = tf.nn.embedding_lookup(q, x[:, 1:2])
    predict = u + target_b_user + target_b_item + tf.matmul(target_q, tf.transpose(target_p, perm=[0, 2, 1]))
    '''

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

    # 最小化方差
    pre_loss = tf.reduce_sum(tf.square(y - predict))

    # 正则项
    l2_weight = tf.constant(l2_weight)
    # l2_target_p = tf.nn.l2_loss(target_p)
    # l2_target_q = tf.nn.l2_loss(target_q)
    # l2_target_bu = tf.nn.l2_loss(target_b_user)
    # l2_target_bi = tf.nn.l2_loss(target_b_item)
    l2_target_p = tf.sqrt(tf.nn.l2_loss(target_p)*2)
    l2_target_q = tf.sqrt(tf.nn.l2_loss(target_q)*2)
    l2_target_bu = tf.sqrt(tf.nn.l2_loss(target_b_user)*2)
    l2_target_bi = tf.sqrt(tf.nn.l2_loss(target_b_item)*2)
    l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q + 1.5 * l2_weight*l2_target_bu + 15 * l2_weight*l2_target_bi

    loss = pre_loss + l2_all
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # tensorboard
    tf.summary.scalar('pre_loss', pre_loss)
    tf.summary.scalar('total_loss', loss)
    tf.summary.histogram('b_user', b_user)
    tf.summary.histogram('b_item', b_item)

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
        for cur_epoch in range(epoch):
            print("current epoch ", cur_epoch)
            for step in range(math.floor(len(x_data)/batch_size)):
                if (step + 1) * batch_size > len(x_data):
                    index = permutation[step * batch_size:]
                else:
                    index = permutation[step * batch_size:(step + 1) * batch_size]
                sess.run(train, feed_dict={x: x_data[index], y: y_data[index]})

            predict_loss = sess.run(pre_loss, feed_dict={x: x_data, y: y_data})
            # total_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})

            current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
            if total_loss - current_loss < 0.1:
                count += 1
                print(" ***")
                if count >= 10:
                    break
            else:
                count = 0
            total_loss = current_loss

            print("pre_loss ", predict_loss)
            print("total_loss ", total_loss)

            rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
            writer.add_summary(rs, cur_epoch)

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

        saveMatrix(k, result_matrix)
        np.savetxt(os.path.join(logPath, "b_user_5"), b_user.eval(session=sess))
        np.savetxt(os.path.join(logPath, "b_item_5"), b_item.eval(session=sess))
        np.savetxt(os.path.join(logPath, "p_5"), p.eval(session=sess))
        np.savetxt(os.path.join(logPath, "q_5"), q.eval(session=sess))

        print(datetime.datetime.now())
        print("finish!")
