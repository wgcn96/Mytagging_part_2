# -*-encoding:utf-8-*-

import tensorflow as tf
import numpy as np
np.random.seed(0)
import random
import time
import datetime
import math
import os

import sys
sys.path.extend(['/home/wangchen/movie tagging/Mytagging_part_2',\
                '/home/wangchen/movie tagging/Mytagging_part_2/SVD_3'])

# 静态文件目录
from SVD_3.static import *
from script.Conf import ConfigParser


def saveMatrix(k, matrix, fileName = None):
    if fileName is None:
        fileName = "resultMatrix_k={}".format(k)
    filePath = os.path.join(output_dir, 'own', fileName)
    np.save(filePath, matrix)


def fillDir(basedir, filename, n):
    full_path = os.path.join(basedir, str(n), filename)
    print(full_path)
    return full_path


if __name__ == '__main__':

    # global variables

    para_file_path = os.path.join(os.getcwd(), 'settings', '0331_5.conf')
    # para_file_path = os.path.join(codedir, 'settings', 'bpr', '0331_5.conf')
    con = ConfigParser(para_file_path)
    res = con.get_config()

    # 聚类簇数量
    cluster_k = int(res['cluster_k'])
    k = int(res['k'])       # 隐向量维度
    epoch = float(res['epoch'])
    l2_weight = float(res['l2_weight'])
    learning_rate = float(res['learning_rate'])
    l2_b_extra = float(res['l2_b_extra'])
    l2_extra = float(res['l2_extra'])
    sample_n = int(res['sample_n'])
    missing_n = int(res['missing_n'])
    logPath = fillDir(log_dir, res['logpath'], missing_n)

    print("Starting Load Rating Matrix")
    print(datetime.datetime.now())

    # 加载四个矩阵
    # rate_matrix  0，1评分矩阵
    # implicit_matrix  隐式反馈矩阵
    # c_matrix  tag之间关联性矩阵
    # theta_matrix  tag之间相似性矩阵

    rate_matrix = np.load(fillDir(data_dir, 'newcom_matrix_sample.npy', missing_n))

    shape = rate_matrix.shape
    movie_count = shape[0]
    tag_count = shape[1]
    implicit_matrix_path = os.path.join(data_dir, "review_implicit_matrix.npy")
    implicit_matrix = np.load(implicit_matrix_path).astype(np.float32)         # 隐式反馈矩阵

    community_cluster_matrix = np.load(os.path.join(data_dir, "fuzzy_U.npy")).astype(np.float32)         # 模糊聚类矩阵
    community_cluster_index_matrix = np.load(os.path.join(data_dir, "final_U.npy")).astype(np.float32)     # 模糊聚类指示矩阵（0，1）

    print("Starting Caculate Average Rating u")
    pos_sample = np.where(rate_matrix == 1)
    x_data = np.array(pos_sample).T  # 转化为480000*2的二维
    total_length = x_data.shape[0]
    y_data = np.ones((total_length, 1), dtype=np.int32)
    matrix_length = movie_count * tag_count
    u = float(total_length) / matrix_length
    all_negrate_index_list = []
    for i in range(movie_count):
        all_negrate_index_list.append(np.where(rate_matrix[i] == 0)[0])
    print("load matrix finish.")

    u = tf.constant(u, dtype=tf.float32)
    X = tf.placeholder(tf.int32, [None, 1], name="X")
    Y_i = tf.placeholder(tf.int32, [None, 1], name="Y_i")
    Y_j = tf.placeholder(tf.int32, [None, 1], name="Y_j")

    '''
    p = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'p')).astype(np.float32), name="user_matrix")
    q = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'q')).astype(np.float32), name="item_matrix")
    b_user = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'b_user')).reshape(-1, 1).astype(np.float32), name="b_user")
    b_item = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'b_item')).reshape(-1, 1).astype(np.float32), name="b_item")
    '''
    b_user = tf.Variable(tf.random_uniform([movie_count, 1], -1e-2, 1e-2, dtype=tf.float32), name="b_user")
    b_item = tf.Variable(tf.random_uniform([tag_count, 1], -1e-2, 1e-2, dtype=tf.float32), name="b_item")
    p = tf.Variable(tf.random_uniform([movie_count, k], -0.5, 0.5, dtype=tf.float32), name="user_matrix")
    q = tf.Variable(tf.random_uniform([tag_count, k], -0.5, 0.5, dtype=tf.float32), name="item_matrix")
    community_cluster_y = tf.Variable(tf.random_uniform([cluster_k, k],  -1e-4, 1e-4, dtype=tf.float32), name="community_cluster_y")
    c_matrix = tf.Variable(tf.random_uniform([tag_count, tag_count],  -1e-2, 1e-2, dtype=tf.float32), name="c_matrix")

    with tf.name_scope('pre') as scope:
        x = X[:, 0]
        y_i = Y_i[:, 0]
        y_j = Y_j[:, 0]
        target_b_user = tf.nn.embedding_lookup(b_user, x)
        target_b_item_i = tf.nn.embedding_lookup(b_item, y_i)
        target_b_item_j = tf.nn.embedding_lookup(b_item, y_j)
        target_p = tf.nn.embedding_lookup(p, x)
        target_q_i = tf.nn.embedding_lookup(q, y_i)
        target_q_j = tf.nn.embedding_lookup(q, y_j)
        target_p = tf.reshape(target_p, [-1, 1, k])
        target_q_i = tf.reshape(target_q_i, [-1, k, 1])
        target_q_j = tf.reshape(target_q_j, [-1, k, 1])

        # theta i 部分
        target_theta_i = tf.nn.embedding_lookup(community_cluster_matrix, y_i)      # theta(i)  （？，聚类簇数量 cluster_k）
        target_theta_sum_i = tf.matmul(target_theta_i, community_cluster_y)                 # （？，cluster_k）* （cluster_k，k）
        target_q_i = target_q_i + tf.reshape(target_theta_sum_i, [-1, k, 1])      # broadcast

        target_theta_j = tf.nn.embedding_lookup(community_cluster_matrix, y_j)      # theta(i)  （？，聚类簇数量 cluster_k）
        target_theta_sum_j = tf.matmul(target_theta_j, community_cluster_y)                 # （？，cluster_k）* （cluster_k，k）
        target_q_j = target_q_j + tf.reshape(target_theta_sum_j, [-1, k, 1])      # broadcast

        bias_i = u + target_b_user + target_b_item_i
        pre_i = tf.matmul(target_p, target_q_i)
        pre_i = tf.reshape(pre_i, [-1, 1])
        pre_i = pre_i + bias_i

        bias_j = u + target_b_user + target_b_item_j
        pre_j = tf.matmul(target_p, target_q_j)
        pre_j = tf.reshape(pre_j, [-1, 1])
        pre_j = pre_j + bias_j

        # Cij 部分
        target_u_implicit_i = tf.nn.embedding_lookup(implicit_matrix, x)     # Ru 行向量（？，2015）
        target_item_in_c_i = tf.nn.embedding_lookup(c_matrix, y_i)    # Ci 行向量（？，2015）
        target_c_row_i = tf.multiply(target_u_implicit_i, target_item_in_c_i)        # 对应元素乘，0的位置为0，1的位置为Cij。相当于对元素按下标1取值。（？，2015）
        sum_c_i = tf.reduce_sum(target_c_row_i, 1, keepdims=True)                                 # 按行求和  （？，1）
        pre_i = pre_i + sum_c_i

        target_u_implicit_j = tf.nn.embedding_lookup(implicit_matrix, x)     # Ru 行向量（？，2015）
        target_item_in_c_j = tf.nn.embedding_lookup(c_matrix, y_j)    # Ci 行向量（？，2015）
        target_c_row_j = tf.multiply(target_u_implicit_j, target_item_in_c_j)        # 对应元素乘，0的位置为0，1的位置为Cij。相当于对元素按下标1取值。（？，2015）
        sum_c_j = tf.reduce_sum(target_c_row_j, 1, keepdims=True)                                 # 按行求和  （？，1）
        pre_j = pre_j + sum_c_j

    # 最小化方差
    with tf.name_scope('loss') as scope:
        # pre_loss = - tf.reduce_mean(tf.log(tf.sigmoid(pre_i - pre_j)))
        # bpr 中，成对损失用的是sigma求和，而不是平均，因此用sum函数
        pre_loss = -tf.reduce_sum(tf.log(tf.sigmoid(pre_i - pre_j)))

        # 正则项
        l2_weight = tf.constant(l2_weight)
        l2_target_p = tf.nn.l2_loss(target_p)*2
        l2_target_q_i = tf.nn.l2_loss(target_q_i)*2
        l2_target_q_j = tf.nn.l2_loss(target_q_j)*2
        l2_target_bu = tf.nn.l2_loss(target_b_user)*2
        l2_target_bi = tf.nn.l2_loss(target_b_item_i)*2
        l2_target_bj = tf.nn.l2_loss(target_b_item_j)*2

        # l2_theta_c
        l2_target_sum_c_i = tf.multiply(target_c_row_i, target_c_row_i)
        l2_target_sum_c_i = tf.reduce_sum(l2_target_sum_c_i)

        l2_target_sum_c_j = tf.multiply(target_c_row_j, target_c_row_j)
        l2_target_sum_c_j = tf.reduce_sum(l2_target_sum_c_j)

        # l2_theta_y
        target_theta_index_i = tf.nn.embedding_lookup(community_cluster_index_matrix, y_i)      # theta(i)  （？，cluster_k）
        target_theta_y_i = tf.matmul(target_theta_index_i, community_cluster_y)                           # （？，k）
        l2_target_theta_y_sum_i = tf.multiply(target_theta_y_i, target_theta_y_i)
        l2_target_theta_y_sum_i = tf.reduce_sum(l2_target_theta_y_sum_i)

        target_theta_index_j = tf.nn.embedding_lookup(community_cluster_index_matrix, y_j)      # theta(i)  （？，cluster_k）
        target_theta_y_j = tf.matmul(target_theta_index_j, community_cluster_y)                           # （？，k）
        l2_target_theta_y_sum_j = tf.multiply(target_theta_y_j, target_theta_y_j)
        l2_target_theta_y_sum_j = tf.reduce_sum(l2_target_theta_y_sum_j)


        l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q_i + l2_weight*l2_target_q_j + l2_b_extra * l2_weight*l2_target_bu + \
                 l2_b_extra * l2_weight*l2_target_bi + l2_b_extra * l2_weight*l2_target_bj + l2_extra * l2_weight*l2_target_sum_c_i + l2_extra * l2_weight*l2_target_sum_c_j + \
                 l2_extra * l2_weight*l2_target_theta_y_sum_i + l2_extra * l2_weight*l2_target_theta_y_sum_j

        loss = pre_loss + l2_all

    current_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate, current_step, 8, 0.95, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=current_step)
    neg_train = optimizer.minimize(loss)

    tf.summary.scalar('pre_loss', pre_loss)
    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

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
        #count = 0
        neg_sample_list = []
        for u in x_data[:, 0]:
            neg_tag_pos = np.random.randint(len(all_negrate_index_list[u]))
            neg_sample_list.append(neg_tag_pos)
        neg_sample_list = np.array(neg_sample_list)
        current_loss = sess.run(loss, feed_dict={X: x_data[:, 0].reshape(-1, 1), Y_i: x_data[:, 1].reshape(-1, 1), Y_j: neg_sample_list.reshape(-1, 1)})
        print("total epoch : {}".format(epoch))
        print("total loss : {}".format(current_loss))

        total_step = int(epoch)

        for step in range(total_step):
            print('current epoch : {}'.format(step))
            neg_sample_list = []
            for u in x_data[permutation][:, 0]:
                neg_tag_pos = np.random.randint(len(all_negrate_index_list[u]))
                neg_sample_list.append(neg_tag_pos)
            neg_sample_list = np.array(neg_sample_list)
            sess.run(train, feed_dict={X: x_data[permutation][:, 0].reshape(-1, 1), Y_i: x_data[permutation][:, 1].reshape(-1, 1), Y_j: neg_sample_list.reshape(-1, 1)})

            predict_loss = sess.run(pre_loss, feed_dict={X: x_data[permutation][:, 0].reshape(-1, 1), Y_i: x_data[permutation][:, 1].reshape(-1, 1), Y_j: neg_sample_list.reshape(-1, 1)})
            total_loss = sess.run(loss, feed_dict={X: x_data[permutation][:, 0].reshape(-1, 1), Y_i: x_data[permutation][:, 1].reshape(-1, 1), Y_j: neg_sample_list.reshape(-1, 1)})
            print("pre_loss ", predict_loss)
            print("total_loss ", total_loss)
            rs = sess.run(merged, feed_dict={X: x_data[permutation][:, 0].reshape(-1, 1), Y_i: x_data[permutation][:, 1].reshape(-1, 1), Y_j: neg_sample_list.reshape(-1, 1)})
            writer.add_summary(rs, step)
            pass  # end for

        # 生成结果矩阵
        result_matrix = np.zeros((movie_count, tag_count), dtype=np.float32)
        for i in range(movie_count):
            rate_index_list = []
            for j in range(tag_count):
                rate_index_list.append([i, j])

            x_pre = np.array(rate_index_list)
            cur_row = sess.run(pre_i, feed_dict={X: x_pre[:, 0].reshape(-1, 1), Y_i: x_pre[:, 1].reshape(-1, 1)})
            result_matrix[i] = cur_row[:, 0]
        result_matrix = result_matrix.reshape(movie_count, tag_count)

        fileName = "bpr_{}".format(missing_n)
        saveMatrix(k, result_matrix, fileName)

    print(datetime.datetime.now())
    print("finish!")

