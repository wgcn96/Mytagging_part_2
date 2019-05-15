# -*-encoding:utf-8 -*-

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

    para_file_path = os.path.join(codedir, 'settings', '0329_wtbatch', '0403_minibatch_5.conf')
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

    another_review_matrix = np.load(fillDir(data_dir, 'another_review_matrix.npy', missing_n))


    shape = rate_matrix.shape
    movie_count = shape[0]
    tag_count = shape[1]
    implicit_matrix_path = os.path.join(data_dir, "review_implicit_matrix.npy")
    implicit_matrix = np.load(implicit_matrix_path).astype(np.float32)         # 隐式反馈矩阵

    community_cluster_matrix = np.load(os.path.join(data_dir, "fuzzy_U.npy")).astype(np.float32)         # 模糊聚类矩阵
    community_cluster_index_matrix = np.load(os.path.join(data_dir, "final_U.npy")).astype(np.float32)     # 模糊聚类指示矩阵（0，1）

    print("Starting Caculate Average Rating u")
    start = time.time()

    movie_list = np.sum(another_review_matrix, 1)
    movie_list_log = np.log2(movie_list)
    movie_list_log /= np.sum(movie_list_log)

    print(np.where(movie_list == 0))
    result_matrix = np.load(os.path.join(data_dir, 'svd_own_0210', 'resultMatrix_k=30.npy'))

