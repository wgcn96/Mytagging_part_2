# -*- coding:utf-8 -*-

"""
获取tag大于20和大于60的电影
"""

import json
import os
import time
import collections
import random

import numpy as np
np.random.seed(0)

from script.static import *
from script.load_implicit_review import loadmovieList


def get_files(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        files
    return files


def get_abundant_movie():
    abundant_movie_dict = {}
    f = open(all_tags_file, encoding='utf-8')
    all_tags_content = json.load(f)
    for movie, tags in all_tags_content.items():
        if len(tags) >= 20:
            abundant_movie_dict[movie] = tags
    print(len(abundant_movie_dict))
    f.close()
    return abundant_movie_dict


def get_60_movie_id(output_file):
    abundant_movie_dict = {}
    f = open(all_tags_file, encoding='utf-8')
    all_tags_content = json.load(f)
    for movie, tags in all_tags_content.items():
        if len(tags) >= 60:
            abundant_movie_dict[movie] = tags
    print(len(abundant_movie_dict))
    f.close()

    output_f = open(output_file, 'w', encoding='utf-8')
    movie_list = abundant_movie_dict.keys()
    for movie in movie_list:
        line = movie + '\n'
        output_f.write(line)
    output_f.close()
    return movie_list


def get_60_movie_line_num():
    _60_movie_id_file_path = os.path.join(output_dir, '60_movie_id.txt')
    _60_movie_pos_file_path = os.path.join(output_dir, '60_movie_pos.txt')
    movie_file = os.path.join(output_dir, 'abundant_movie.txt')

    movieList, movie_dict = loadmovieList(movie_file)
    _60_movie_id_file = open(_60_movie_id_file_path, encoding='utf-8')
    _60_movie_pos_file = open(_60_movie_pos_file_path, 'w', encoding='utf-8')

    while True:
        line = _60_movie_id_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        movie_pos = movie_dict[line]
        w_line = str(movie_pos) + '\n'
        _60_movie_pos_file.write(w_line)

    _60_movie_pos_file.close()
    _60_movie_id_file.close()


def get_detail(abundant_movie_dict):
    movie_order_dict = {}
    tag_order_dict = {}

    movie_order_list = []
    tag_order_list = []

    tags_set = set()
    tags_times = 0
    movie_order = 0
    for movie, tags in abundant_movie_dict.items():
        movie_order_list.append(movie)
        movie_order_dict[movie] = movie_order
        movie_order += 1
        tags_times += tags.__len__()
        tags_set.update(tags)
    print("标签集合大小和打标签次数：", end='')
    print(tags_set.__len__(), tags_times)

    tag_order_list = list(tags_set)

    tag_order = 0
    for item in tag_order_list:
        tag_order_dict[item] = tag_order
        tag_order += 1

    return movie_order_list, tag_order_list, movie_order_dict, tag_order_dict


def abundant_movie_save_file(abundant_movie_dict):
    movie_order_file_path = os.path.join(output_dir, "abundant_movie.txt")
    tag_order_file_path = os.path.join(output_dir, "abundant_tag.txt")
    abundant_movie_file_path = os.path.join(output_dir, 'abundant_movie_file.json')
    movie_order_list, tag_order_list, movie_order_dict, tag_order_dict = get_detail(abundant_movie_dict)

    shape = (len(movie_order_dict), len(tag_order_dict))

    matrix = np.zeros(shape, dtype=np.int32)

    for cur_movie_order, movie in enumerate(movie_order_list):
        tags = abundant_movie_dict[movie]
        for tag in tags:
            cur_tag_order = tag_order_dict[tag]
            matrix[cur_movie_order, cur_tag_order] = 1

    print(np.sum(matrix))

    matrix_path = os.path.join(output_dir, "abundant_movie_matrix.npy")
    np.save(matrix_path, matrix)
    movie_order_file = open(movie_order_file_path, 'w', encoding='utf-8')
    tag_order_file = open(tag_order_file_path, 'w', encoding='utf-8')
    abundant_movie_file = open(abundant_movie_file_path, 'w', encoding='utf-8')

    for movie in movie_order_list:
        movie_order_file.write(movie+'\n')
    for tag in tag_order_list:
        tag_order_file.write(tag+'\n')

    json.dump(abundant_movie_dict, abundant_movie_file, indent=4, sort_keys=True, ensure_ascii=False)

    print("finish!")


# 统计一个目录下json文件数，以及json中array数据的条数
def count_all_files(root_dir, files):
    file_count = 0      # 共计打开对少个文件
    count = 0       # 共计过少条记录
    for file in files:
        file_root = os.path.join(root_dir,  file)
        f = open(file_root, encoding='utf-8')
        content = json.load(f)
        data = content["data"]  # data是一个list，每条记录为一个 dict
        count += len(data)
        if len(data):
            file_count += 1
        f.close()
    return count, file_count


# 文件树操作，打印所有文件
def walkdir(dirname):
    try:
        ls = os.listdir(dirname)
    except:
        print('access deny')
    else:
        for l in ls:
            temp = os.path.join(dirname, l)
            if os.path.isdir(temp):
                walkdir(temp)
            else:
                print(temp)


# 目录中获取一级子目录下列表  与  get_files() 连用
def walk_into_dir(dirname):
    dir_list = []
    ls = os.listdir(dirname)
    for item in ls:
        temp = os.path.join(dirname, item)
        if os.path.isdir(temp):
            # temp += '\\'
            dir_list.append(temp)
    return dir_list


# 取所有电影的评分作为字典
def get_movie_rate_dict():
    movie_rate_dict = {}
    files = get_files(all_movies_dir)
    for file in files:
        # print("current file", file)
        file_root = os.path.join(all_movies_dir,  file)
        f = open(file_root, encoding='utf-8')
        content = json.load(f)
        data = content["data"]  # data是一个list，每条记录为一个 dict
        for record in data:
            cur_movie = record['id']
            movie_rate_dict[cur_movie] = float(file[:3])
        f.close()
    return movie_rate_dict


def get_abundant_movie_y(movie_rate_dict):
    y = np.zeros((4311, 1), np.float32)

    movie_order_file_path = os.path.join(output_dir, "abundant_movie.txt")
    # abundant_movie_rate_file_path = os.path.join(output_dir, 'abundant_movie_rates.json')
    result_dict = {}
    movie_order_file = open(movie_order_file_path, 'r', encoding='utf-8')
    count = -1
    while True:
        line = movie_order_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        count += 1
        movie = line.split(" ")[0]
        cur_movie_rate = movie_rate_dict.get(movie)
        if cur_movie_rate:
            y[count] = cur_movie_rate
        result_dict[movie] = y[count]
    return y, result_dict


def getSample(vector, n, condition):
    """
    采样函数
    :param vector: 行向量
    :param n: 采样本数量
    :param condition:  采样条件
    :return: 采样list转化为np array
    """
    '''
    result_list = []
    length = len(vector)
    while len(result_list) < n:
        pos = random.randint(0, length)
        if vector[pos] == condition and pos not in result_list:
            result_list.append(pos)
    print(result_list)
    return result_list
    '''
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


def matrix_sample(n):
    abundant_movie_file_path = os.path.join(output_dir, 'abundant_movie_matrix.npy')
    matrix = np.load(abundant_movie_file_path)
    comprehensive_matrix_file_path = os.path.join(output_dir, 'comprehensive_matrix.npy')
    comprehensive_matrix = np.load(comprehensive_matrix_file_path)
    result_matrix_file_path = os.path.join(output_dir, 'comprehensive_index_matrix.npy')
    result_matrix = np.load(result_matrix_file_path)

    _60_movie_pos_file_path = os.path.join(output_dir, '60_movie_pos.txt')
    _60_movie_pos_file = open(_60_movie_pos_file_path, encoding='utf-8')
    pos_list = []
    while True:
        line = _60_movie_pos_file.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        pos_list.append(int(line))
    _60_movie_pos_file.close()
    print(len(pos_list))

    for i in pos_list:
        result_list = getSample(matrix[i], n, condition=1)
        comprehensive_matrix[i][result_list] = -1
        result_matrix[i][result_list] = -1

    comprehensive_matrix_file_path = os.path.join(output_dir, 'comprehensive_matrix_sample.npy')
    np.save(comprehensive_matrix_file_path, comprehensive_matrix)
    result_matrix_file_path = os.path.join(output_dir, 'comprehensive_index_matrix_sample.npy')
    np.save(result_matrix_file_path, result_matrix)
    return comprehensive_matrix, result_matrix

if __name__ == "__main__":
    # (4311, 1506)

    '''
    # 保存y
    abundant_movie_dict = get_abundant_movie()
    movie_rate_dict = get_movie_rate_dict()
    y, result_dict = get_abundant_movie_y(movie_rate_dict)
    abundant_movie_rate_file_path = os.path.join(output_dir, 'abundant_movie_rates.json')
    abundant_movie_rate_npfile_path = os.path.join(output_dir, 'abundant_movie_rates.npy')

    np.save(abundant_movie_rate_npfile_path, y)
    file = open(abundant_movie_rate_file_path, encoding='utf-8', mode='w')
    json.dump(movie_rate_dict, file, indent=4, sort_keys=True, ensure_ascii=False)
    file.close()
    '''

    '''
    # 保存矩阵
    abundant_movie_save_file(abundant_movie_dict)
    abundant_movie_file_path = os.path.join(output_dir, 'abundant_movie_matrix.npy')
    implicit_movie_file_path = os.path.join(output_dir, 'review_implicit_matrix.npy')
    matrix = np.load(abundant_movie_file_path)
    matrix_2 = np.load(implicit_movie_file_path)
    result_matrix = matrix_2 + 2 * matrix
    print(result_matrix.shape)
    result_matrix_file_path = os.path.join(output_dir, 'comprehensive_matrix.npy')
    np.save(result_matrix_file_path, result_matrix)
    '''

    '''
    2 + 1
    comprehensive_matrix_file_path = os.path.join(output_dir, 'comprehensive_matrix.npy')
    comprehensive_matrix = np.load(comprehensive_matrix_file_path)
    result_matrix_file_path = os.path.join(output_dir, 'comprehensive_index_matrix.npy')
    result_matrix = np.where(comprehensive_matrix != 0, np.ones((4311, 1506), dtype=np.int32), np.zeros((4311, 1506), dtype=np.int32))
    np.save(result_matrix_file_path, result_matrix)
    '''

    '''
    
    _60_movie_id_file = os.path.join(output_dir, '60_movie_id.txt')
    get_60_movie_id(_60_movie_id_file)
    get_60_movie_line_num()
    '''

    comprehensive_matrix, result_matrix = matrix_sample(10)
