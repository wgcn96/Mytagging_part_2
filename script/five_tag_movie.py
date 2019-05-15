# -*- coding:utf-8 -*-

"""
获取tag大于5的电影
"""

import json
import os
import time
import collections

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


def loadTagList(tagFile):
    tag_dict = {}       # 反取，取下标
    tagList = []        # 正取，取tag名称
    f = open(tagFile, 'r', encoding='utf-8')
    count = -1
    while True:
        count += 1
        line = f.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        tag = line.split(" ")[0]
        tagList.append(tag)
        tag_dict[tag] = count
    f.close()
    print(" tag List len: {}".format(len(tagList)))

    return tagList, tag_dict


def loadmovieList(movieFile):
    movie_dict = {}       # 反取，取下标
    movieList = []        # 正取，取movie名称
    f = open(movieFile, 'r', encoding='utf-8')
    count = -1
    while True:
        count += 1
        line = f.readline()
        line = line.strip()

        if line == "" or line is None:
            break

        movie = line.split(" ")[0]
        movieList.append(movie)
        movie_dict[movie] = count
    f.close()
    print(" movie List len: {}".format(len(movieList)))

    return movieList, movie_dict


def get_abundant_movie():
    abundant_movie_dict = {}
    f = open(all_tags_file, encoding='utf-8')
    all_tags_content = json.load(f)
    for movie, tags in all_tags_content.items():
        if len(tags) >= 5:
            abundant_movie_dict[movie] = tags
    print(len(abundant_movie_dict))
    f.close()
    return abundant_movie_dict


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
    movie_order_file_path = os.path.join(output_2_dir, "movie.txt")
    tag_order_file_path = os.path.join(output_2_dir, "tag.txt")
    abundant_movie_file_path = os.path.join(output_2_dir, 'movie_file.json')
    movie_order_list, tag_order_list, movie_order_dict, tag_order_dict = get_detail(abundant_movie_dict)

    shape = (len(movie_order_dict), len(tag_order_dict))

    matrix = np.zeros(shape, dtype=np.int32)

    for cur_movie_order, movie in enumerate(movie_order_list):
        tags = abundant_movie_dict[movie]
        for tag in tags:
            cur_tag_order = tag_order_dict[tag]
            matrix[cur_movie_order, cur_tag_order] = 1

    print(np.sum(matrix))

    matrix_path = os.path.join(output_2_dir, "movie_matrix.npy")
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
    y = np.zeros((19004, 1), np.float32)

    movie_order_file_path = os.path.join(output_2_dir, "movie.txt")
    # abundant_movie_rate_file_path = os.path.join(output_2_dir, 'abundant_movie_rates.json')
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


def get_abundant_movie_pos_list():
    abundant_movie_pos_list = []

    abundant_movie_file = os.path.join(output_dir, 'abundant_movie.txt')
    movie_file = os.path.join(output_2_dir, 'movie.txt')
    result_file = os.path.join(output_dir, 'abundant_movie_pos.npy')

    movieList, movie_dict = loadmovieList(movie_file)
    abundant_movieList, _ = loadmovieList(abundant_movie_file)

    for item in abundant_movieList:
        pos = movie_dict[item]
        abundant_movie_pos_list.append(pos)

    print("total length {}".format(len(abundant_movie_pos_list)))

    abundant_movie_pos_list = np.asarray(abundant_movie_pos_list)
    np.save(result_file, abundant_movie_pos_list)

    return abundant_movie_pos_list


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
    abundant_movie_file_path = os.path.join(output_2_dir, 'movie_matrix.npy')
    matrix = np.load(abundant_movie_file_path)
    comprehensive_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_matrix.npy')
    comprehensive_matrix = np.load(comprehensive_matrix_file_path)
    result_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_index_matrix.npy')
    result_matrix = np.load(result_matrix_file_path)

    pos_list = np.load(os.path.join(output_2_dir, 'abundant_movie_pos.npy'))

    for i in pos_list:
        result_list = getSample(matrix[i], n, condition=1)
        comprehensive_matrix[i][result_list] = -1
        result_matrix[i][result_list] = -1

    comprehensive_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_matrix_sample.npy')
    np.save(comprehensive_matrix_file_path, comprehensive_matrix)
    result_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_index_matrix_sample.npy')
    np.save(result_matrix_file_path, result_matrix)
    return comprehensive_matrix, result_matrix


def generate_another_matrix():
    """
    用影评中出现的标签（包括非关键词标签）生成的矩阵
    :return:
    """
    counter = 0
    movie_file = os.path.join(output_2_dir, 'movie.txt')
    movieList, movie_dict = loadmovieList(movie_file)
    tag_file = os.path.join(output_2_dir, 'tag.txt')
    tagList, tag_dict = loadTagList(tag_file)
    result_matrix = np.zeros((19004, 1896), dtype=np.int32)
    another_tags_file = os.path.join(relative_data_dir, 'Special.json')
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
    print(counter)
    np.save(os.path.join(output_2_dir, 'another_review_matrix.npy'), result_matrix)
    return result_matrix


def new_combine(n = 5):
    """
    用影评中出现的标签做合并
    :return:
    """
    another_matrix = np.load(os.path.join(output_2_dir, 'another_review_matrix.npy'))
    ori_matrix = np.load(os.path.join(output_2_dir, str(n), 'newcom_matrix_sample.npy'))
    result_matrix_path = os.path.join(output_2_dir, str(n), 'another_review_matrix.npy')
    result_matrix = np.where(ori_matrix == 0, another_matrix, ori_matrix)
    result_matrix = result_matrix.astype(dtype=np.int32)
    np.save(result_matrix_path, result_matrix)

if __name__ == "__main__":
    # (19004, 1896)

    # 保存y
    # abundant_movie_dict = get_abundant_movie()
    # movie_rate_dict = get_movie_rate_dict()

    # abundant_movie_save_file(abundant_movie_dict)

    # y, result_dict = get_abundant_movie_y(movie_rate_dict)
    # abundant_movie_rate_file_path = os.path.join(output_2_dir, 'movie_rates.json')
    # abundant_movie_rate_npfile_path = os.path.join(output_2_dir, 'movie_rates.npy')
    #
    # np.save(abundant_movie_rate_npfile_path, y)
    # file = open(abundant_movie_rate_file_path, encoding='utf-8', mode='w')
    # json.dump(movie_rate_dict, file, indent=4, sort_keys=True, ensure_ascii=False)
    # file.close()

    '''
    # 保存矩阵
    abundant_movie_file_path = os.path.join(output_2_dir, 'movie_matrix.npy')
    implicit_movie_file_path = os.path.join(output_2_dir, 'review_implicit_matrix.npy')
    matrix = np.load(abundant_movie_file_path)
    matrix_2 = np.load(implicit_movie_file_path)
    result_matrix = matrix_2 + 2 * matrix
    print(result_matrix.shape)
    result_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_matrix.npy')
    np.save(result_matrix_file_path, result_matrix)

    comprehensive_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_matrix.npy')
    comprehensive_matrix = np.load(comprehensive_matrix_file_path)
    result_matrix_file_path = os.path.join(output_2_dir, 'comprehensive_index_matrix.npy')
    result_matrix = np.where(comprehensive_matrix != 0, np.ones((19004, 1896), dtype=np.int32), np.zeros((19004, 1896), dtype=np.int32))
    np.save(result_matrix_file_path, result_matrix)
    '''

    '''
    _60_movie_id_file = os.path.join(output_2_dir, '60_movie_id.txt')
    get_60_movie_id(_60_movie_id_file)
    get_60_movie_line_num()
    '''

    '''
    abundant_movie_pos_list = get_abundant_movie_pos_list()
    '''

    '''
    comprehensive_matrix, result_matrix = matrix_sample(10)
    '''

    result_matrix = generate_another_matrix()
    # new combine
    # n = 20
    # new_combine(n)
