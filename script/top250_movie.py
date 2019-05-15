# -*- coding:utf-8 -*-

"""
top250电影
"""

import json
import os
import time
import collections
import random

import numpy as np
np.random.seed(0)

from script.static import *
from script.load_implicit_review import loadmovieList, loadTagList


def get_top250_movie_id():
    top250_movie_pos_file_path = os.path.join(relative_data_dir, 'top250.txt')
    top250_movie_pos_file = open(top250_movie_pos_file_path, encoding='utf-8')
    id_list = []
    while True:
        line = top250_movie_pos_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        id_list.append(line)
    top250_movie_pos_file.close()
    return id_list


def getSample(vector, n, condition):
    """
    采样函数
    :param vector: 行向量
    :param n: 采样本数量
    :param condition:  采样条件
    :return: 采样list转化为np array
    """
    candidate_list = []
    result_list = []
    for pos, item in enumerate(vector):
        if item == condition:
            candidate_list.append(pos)
    length = len(candidate_list)

    if length < n:
        print("采样数量大于最大采样值")
        return -1

    permatation = np.random.permutation(length)
    result_list = np.asarray(candidate_list)[permatation[:n]]
    # print(result_list)
    return result_list


def getSampleNew(vector, n, condition):
    """
    采样函数，采样数量n做了soft软条件
    :param vector: 行向量
    :param n: 采样本数量
    :param condition:  采样条件
    :return: 采样list转化为np array
    """

    candidate_list = []
    result_list = []
    for pos, item in enumerate(vector):
        if item == condition:
            candidate_list.append(pos)
    length = len(candidate_list)

    if length < n:
        n = length
        print(length)

    permatation = np.random.permutation(length)
    result_list = np.asarray(candidate_list)[permatation[:n]]
    # print(result_list)
    return result_list


def matrix_sample(n):
    """
    矩阵采样函数，并完再采
    :param n: 样本数量n
    :return: 矩阵的采样结果
    """
    abundant_movie_file_path = os.path.join(output_2_dir, 'movie_matrix.npy')
    matrix = np.load(abundant_movie_file_path)
    comprehensive_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_matrix.npy')
    comprehensive_matrix = np.load(comprehensive_matrix_file_path)
    result_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_index_matrix.npy')
    result_matrix = np.load(result_matrix_file_path)

    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))

    for i in pos_list:
        result_list = getSample(matrix[i], n, condition=1)
        comprehensive_matrix[i][result_list] = -1
        result_matrix[i][result_list] = -1

    comprehensive_matrix_file_path = os.path.join(output_2_dir, str(n), 'comprehensive_matrix_sample.npy')
    np.save(comprehensive_matrix_file_path, comprehensive_matrix)
    result_matrix_file_path = os.path.join(output_2_dir, str(n), 'comprehensive_index_matrix_sample.npy')
    np.save(result_matrix_file_path, result_matrix)
    return comprehensive_matrix, result_matrix


def matrix_sample_new(n):
    """
    矩阵采样函数，采完做并
    :param n: 样本数量n
    :return: 矩阵的采样结果
    """
    matrix = np.load(os.path.join(output_2_dir, 'movie_matrix.npy'))

    review_implicit_matrix_path = os.path.join(output_2_dir, "review_implicit_matrix.npy")
    review_implicit_matrix = np.load(review_implicit_matrix_path)

    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))

    print(np.sum(matrix))
    for i in pos_list:
        result_list = getSample(matrix[i], n, condition=1)
        matrix[i][result_list] = -1
    print(np.sum(matrix))
    print(np.sum(review_implicit_matrix))

    matrix = np.where(review_implicit_matrix == 1, review_implicit_matrix, matrix)

    print(np.sum(matrix))
    result_matrix_file_path = os.path.join(output_2_dir, str(n), 'newcom_matrix_sample.npy')
    np.save(result_matrix_file_path, matrix)

    return matrix


def matrix_sample_leave_ten(review_implicit_matrix):
    """
    矩阵采样函数
    :param n: 样本数量n
    :return: 矩阵的采样结果
    """
    matrix = np.load(os.path.join(output_2_dir, 'movie_matrix.npy'))
    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))
    print(np.sum(matrix))
    count = 0
    for i in pos_list:
        sample_list = np.where(matrix[i] ==1)[0]
        permatation = np.random.permutation(len(sample_list))
        result_list = np.asarray(sample_list)[permatation[10:]]
        matrix[i][result_list] = -1
        count += len(result_list)

    print(count)
    result_matrix_file_path = os.path.join(output_2_dir, 'com_matrix_sample_top250_ten.npy')
    np.save(result_matrix_file_path, matrix)
    print(np.sum(matrix))
    matrix = np.where(review_implicit_matrix == 1, review_implicit_matrix, matrix)

    print(np.sum(matrix))
    result_matrix_file_path = os.path.join(output_2_dir, 'new_matrix_sample_top250_ten.npy')
    np.save(result_matrix_file_path, matrix)

    return matrix


def ori_matrix_sample(n, neg_mark):
    """
    原始电影标签矩阵采样，生成采样后的矩阵用于验证review的效果
    :param n: 采样数量
    :param neg_mark: 负采样标记
    :return: 采样后的矩阵
    """
    matrix = np.load(os.path.join(output_2_dir, 'movie_matrix.npy'))
    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))
    print('ori matrix sum: {}'.format(np.sum(matrix)))
    if n == 100:
        count = 0
        for i in pos_list:
            sample_list = np.where(matrix[i] == 1)[0]
            permatation = np.random.permutation(len(sample_list))
            result_list = np.asarray(sample_list)[permatation[10:]]
            matrix[i][result_list] = neg_mark
            count += len(result_list)

    else:
        for i in pos_list:
            result_list = getSample(matrix[i], n, condition=1)
            matrix[i][result_list] = neg_mark
    result_matrix_file_path = os.path.join(output_2_dir, str(n), 'raw_matrix_sample_{}.npy'.format(n))
    np.save(result_matrix_file_path, matrix)
    print('revise matrix sum: {}'.format(np.sum(matrix)))
    return matrix


def leave_out_tag(n, movie_list, tag_list):
    result_dict = {}
    matrix_path = os.path.join(output_2_dir, str(n), 'ori_matrix_sample_{}.npy'.format(n))
    matrix = np.load(matrix_path)
    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))
    for pos in pos_list:
        movie = movie_list[pos]
        result_dict[movie] = []
        cur_row = matrix[pos]
        neg_list = np.where(cur_row == -1)[0]
        for neg in neg_list:
            result_dict[movie].append(tag_list[neg])
    result_path = os.path.join(output_2_dir, str(n), 'leave_out_tag_{}.json'.format(n))
    f = open(result_path, encoding='utf-8', mode='w')
    json.dump(result_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
    f.close()


def generate_another_matrix():
    """
    极度缺失标签情况下，每个电影取22条影评，生成另一个影评矩阵
    :return: 22条影评的review_matrix
    """
    # matrix = np.where(review_implicit_matrix == 1, review_implicit_matrix, matrix)
    counter = 0
    movie_file = os.path.join(output_2_dir, 'movie.txt')
    movieList, movie_dict = loadmovieList(movie_file)
    tag_file = os.path.join(output_2_dir, 'tag.txt')
    tagList, tag_dict = loadTagList(tag_file)
    result_matrix = np.zeros((19004, 1896), dtype=np.int32)
    another_tags_file = os.path.join(relative_data_dir, 'Top250_Review_tag.json')
    file = open(another_tags_file, encoding='utf-8')
    all_tags_content = json.load(file)
    for movie, tags in all_tags_content.items():
        movie_pos = movie_dict.get(movie, -1)
        if movie_pos == -1:
            continue
        for tag in tags:
            tag_pos = tag_dict.get(tag, -1)
            if tag_pos != -1:
                result_matrix[movie_pos][tag_pos] = 1
                counter += 1

    review_implicit_matrix_path = os.path.join(output_2_dir, "review_implicit_matrix.npy")
    review_implicit_matrix = np.load(review_implicit_matrix_path)

    pos_list = np.load(os.path.join(output_2_dir, 'top250_movie_pos.npy'))
    for i in range(result_matrix.shape[0]):
        if i in pos_list:
            continue
        result_matrix[i] = review_implicit_matrix[i]

    return result_matrix


def get_abundant_movie_pos_list(movie_list, result_file):
    """
    top250 电影在19000部电影中的位置，并保存文件
    :param movie_list: top250的id
    :param result_file: 保存结果
    :return: 保存的结果同时返回为pos-list
    """
    result_pos_list = []

    movie_file = os.path.join(output_2_dir, 'movie.txt')
    full_list, full_dict = loadmovieList(movie_file)

    for item in movie_list:
        if item in full_dict.keys():
            pos = full_dict[item]
            result_pos_list.append(pos)
        else:
            print(item)

    print("total length {}".format(len(result_pos_list)))

    result_pos_list = np.asarray(result_pos_list)
    np.save(result_file, result_pos_list)

    return result_pos_list


if __name__ == "__main__":

    # top250_movie = get_top250_movie_id()      # top250电影的id

    # abundant_movie_pos_list = get_abundant_movie_pos_list(top250_movie, result_file)  # 获取top250电影的pos列表
    # pos_file = os.path.join(output_2_dir, 'top250_movie_pos.npy')
    # pos_list = np.load(pos_file)

    # n = 20
    # result_matrix = matrix_sample_new(n)      # 并完再采样的函数

    # review_matrix = generate_another_matrix()       # 用于top250电影随机取22条影评，只保留10个标签生成极度缺失ground-truth
    # a = matrix_sample_leave_ten(review_matrix)

    # ori_matrix_sample(5, -1)        # 原始电影标签矩阵采样，生成采样后的矩阵用于验证review的效果
    # ori_matrix_sample(10, -1)
    # ori_matrix_sample(15, -1)
    # ori_matrix_sample(20, -1)
    # ori_matrix_sample(100, -1)


    '''
    ori_matrix_sample(5, 0)        # 原始电影标签矩阵采样，生成采样后的矩阵用于验证review的效果，标记为0
    ori_matrix_sample(10, 0)
    ori_matrix_sample(15, 0)
    ori_matrix_sample(20, 0)
    ori_matrix_sample(100, 0)
    matrix = np.load(os.path.join(output_2_dir, '100', 'raw_matrix_sample_100.npy'))
    '''

    tagFile = os.path.join(output_2_dir, 'tag.txt')
    tag_list, _ = loadTagList(tagFile)
    movieFile = os.path.join(output_2_dir, 'movie.txt')
    movie_list, _ = loadmovieList(movieFile)
    for n in range(5, 25, 5):
        leave_out_tag(n, movie_list, tag_list)

    n = 100
    leave_out_tag(n, movie_list, tag_list)
