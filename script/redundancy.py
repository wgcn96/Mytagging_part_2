# -*- coding: UTF-8 -*-

import numpy as np
import os

from script.static import *


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


def load():
    path = os.path.join(output_dir, 'abundant_movie_matrix.npy')
    abundant_matrix = np.load(path)
    return abundant_matrix


def get_redundancy_matrix(abundant_matrix):
    shape = abundant_matrix.shape

    redundancy_matrix = np.zeros((shape[0], shape[0]), dtype=np.float32)

    count = 0
    for i in range(shape[0]):
        for j in range(i, shape[0]):
            print('current line {}'.format(i))
            if i == j:
                redundancy_matrix[i,j] = 1
                continue
            vec_i = abundant_matrix[i]
            vec_j = abundant_matrix[j]
            cos = cos_sim(vec_i, vec_j)
            redundancy_matrix[i, j] = cos
            redundancy_matrix[j, i] = cos
            count += 1
    print('total values {}'.format(count))
    return redundancy_matrix


if __name__ == '__main__':
    abundant_matrix = load()
    redundancy_matrix = get_redundancy_matrix(abundant_matrix)
    redundancy_matrix_file_path = os.path.join(output_dir, 'redundancy_matrix_2.npy')
    np.save(redundancy_matrix_file_path, redundancy_matrix)
