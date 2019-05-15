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

    para_file_path = os.path.join(os.getcwd(), 'settings', '0411_component.conf')
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

    # another_review_matrix = np.load(os.path.join(data_dir, 'another_review_matrix.npy'))

    shape = rate_matrix.shape
    movie_count = shape[0]
    tag_count = shape[1]

    print("Starting Caculate Average Rating u")

    movie_list = np.sum(rate_matrix, 1)
    # movie_list_log = np.log2(movie_list)
    movie_list_log = movie_list / np.sum(movie_list)

    print(np.where(movie_list == 0))

    pos_sample = np.where(rate_matrix == 1)

    x_data = np.array(pos_sample).T  # 转化为480000*2的二维
    total_length = x_data.shape[0]
    y_data = np.ones((total_length, 1), dtype=np.int32)
    matrix_length = movie_count * tag_count
    u = float(total_length) / matrix_length
    all_rate_index_list = []
    all_rate_list = []
    for i in range(movie_count):
        for j in range(tag_count):
            all_rate_index_list.append([i, j])
            all_rate_list.append(rate_matrix[i][j])
    all_x_data = np.array(all_rate_index_list)
    all_y_data = np.array(all_rate_list)[:, None]
    print("load matrix finish.")

    u = tf.constant(u, dtype=tf.float32)
    x = tf.placeholder(tf.int32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="Y")

    p = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'p')).astype(np.float32), name="user_matrix")
    q = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'q')).astype(np.float32), name="item_matrix")
    b_user = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'b_user')).reshape(-1, 1).astype(np.float32), name="b_user")
    b_item = tf.Variable(np.loadtxt(os.path.join(log_dir, 'svd_0102', 'b_item')).reshape(-1, 1).astype(np.float32), name="b_item")

    with tf.name_scope('pre') as scope:
        movie_pos = x[:, 0]
        tag_pos = x[:, 1]
        target_b_user = tf.nn.embedding_lookup(b_user, movie_pos)
        target_b_item = tf.nn.embedding_lookup(b_item, tag_pos)
        target_p = tf.nn.embedding_lookup(p, movie_pos)
        target_q = tf.nn.embedding_lookup(q, tag_pos)
        target_p = tf.reshape(target_p, [-1, 1, k])
        target_q = tf.reshape(target_q, [-1, k, 1])

        bias = u + target_b_user + target_b_item
        p_e = tf.matmul(target_p, target_q)
        p_e = tf.reshape(p_e, [-1, 1])
        predict = p_e + bias


    # 最小化方差
    with tf.name_scope('loss') as scope:
        pre_loss = tf.reduce_sum(tf.square(y - predict))

        # 正则项
        l2_weight = tf.constant(l2_weight)
        l2_target_p = tf.nn.l2_loss(target_p)*2
        l2_target_q = tf.nn.l2_loss(target_q)*2
        l2_target_bu = tf.nn.l2_loss(target_b_user)*2
        l2_target_bi = tf.nn.l2_loss(target_b_item)*2

        # l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q + l2_b_extra * l2_weight*l2_target_bu + l2_b_extra * l2_weight*l2_target_bi + l2_extra * l2_weight*l2_target_sum_c + l2_extra * l2_weight*l2_target_theta_y_sum    # ori
        l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q + l2_b_extra * l2_weight*l2_target_bu + l2_b_extra * l2_weight*l2_target_bi
        # l2_all = l2_weight*l2_target_p + l2_weight*l2_target_q + l2_b_extra * l2_weight*l2_target_bu + l2_b_extra * l2_weight*l2_target_bi + l2_extra * l2_weight*l2_target_sum_c + l2_extra * l2_weight*l2_target_theta_y_sum    # y

        loss = pre_loss + l2_all

    current_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate, current_step, 100000, 0.99, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=current_step)
    neg_train = optimizer.minimize(loss)

    tf.summary.scalar('pre_loss', pre_loss)
    tf.summary.scalar('total_loss', loss)
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

        permutation = np.random.permutation(total_length)
        total_loss = 100000000000
        #count = 0
        current_loss = sess.run(loss, feed_dict={x: x_data[:total_length], y: y_data[:total_length]})
        print("total epoch : {}".format(epoch))
        print("total loss : {}".format(current_loss))

        for cur_epoch in range(int(epoch)):
            print('current epoch : {}'.format(cur_epoch))
            
            batch_size = int(440000/20)
            for step in range(math.floor(total_length/batch_size)):
                if (step + 1) * batch_size > total_length:
                    index = permutation[(step * batch_size):]
                else:
                    index = permutation[(step * batch_size): ((step + 1) * batch_size)]
                sess.run(train, feed_dict={x: x_data[index], y: y_data[index]})

                neg_num = sample_n * len(index)
                movie_index_choice = np.random.choice(movie_count, size=neg_num, p=movie_list_log)
                neg_index = []
                for j in range(neg_num):
                    while True:
                        movie_index = movie_index_choice[j]
                        tag_index = np.random.randint(tag_count)
                        if rate_matrix[movie_index][tag_index] == 0:
                            neg_index.append(movie_index * tag_count + tag_index)
                            break
                sess.run(neg_train, feed_dict={x: (all_x_data[neg_index]).reshape(-1, 2),
                                               y: (all_y_data[neg_index]).reshape(-1, 1)})
                pass  # end positive for-loop

            predict_loss = sess.run(pre_loss, feed_dict={x: x_data, y: y_data})
            total_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
            print("pre_loss ", predict_loss)
            print("total_loss ", total_loss)
            rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
            writer.add_summary(rs, cur_epoch)
            pass  # end for

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

        fileName = "unfixed_user_wtbatch{}_{}_componentall".format(missing_n, sample_n)
        # fileName = "unfixed_user_less_log"
        saveMatrix(k, result_matrix, fileName)

    print(datetime.datetime.now())
    print("finish!")

