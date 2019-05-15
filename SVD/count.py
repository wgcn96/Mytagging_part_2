"""
原始数据的统计函数和脚本
"""

import json
import os
import time
import collections

import numpy as np

from static import *


f = open(all_tags_file, encoding='utf-8')
all_tags_content = json.load(f)
count_all_tagged_movies = len(all_tags_content)


def get_files(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        files
    return files


files = get_files(all_movies_dir)


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


count_all_movies, _ = count_all_files(all_movies_dir, files)
print("有标签的电影和所有电影：", end='')
print(count_all_tagged_movies, count_all_movies)
print()

movie_order = 0
tag_order = 0
movie_order_dict = {}
movie_list = []
tag_order_dict = collections.OrderedDict()

tags_set = set()
tags_times = 0
for movie, tags in all_tags_content.items():
    tags_times += tags.__len__()
    tags_set.update(tags)
print("标签集合大小和打标签次数：", end='')
print(tags_set.__len__(), tags_times)
print()

for tag in tags_set:
    tag_order_dict[tag] = tag_order
    tag_order += 1


user_item_matrix = np.zeros([count_all_movies, tags_set.__len__()], dtype=np.int32)

for movie, tags in all_tags_content.items():
    movie_order_dict[movie] = movie_order       # 正取
    movie_list.append(movie)            # 反取
    for tag in tags:
        tag_pos = tag_order_dict[tag]
        user_item_matrix[movie_order, tag_pos] += 1
    movie_order += 1


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


# 此段大量文件读写代码不可执行，只记住结果即可
'''
start = time.time()
review_dir = walk_into_dir(movie_review_dir)
count_reviewed_movies = 0
reviews = 0
for root_dir in review_dir:
    files = get_files(root_dir)
    count = count_all_files(root_dir, files)
    reviews += count[0]
    if count[1]:
        count_reviewed_movies += count[1]
print(count_reviewed_movies, reviews)
print(time.time() - start)
# 运行结果： 36540 1691793   # 长评的电影3万6，共有影评17万
'''


# 一个评分目录下，有影评的电影是否都是有标签的，遍历dict
def check(root_dir, files):
    check_count = 0
    file_count = 0
    for file in files:
        file_root = os.path.join(root_dir,  file)
        f = open(file_root, encoding='utf-8')
        content = json.load(f)
        file_count += 1
        data = content["data"]  # data是一个list，每条记录为一个 dict
        if len(data):
            if os.path.splitext(file)[0] not in movie_order_dict.keys():
                check_count += 1
        f.close()
    return file_count, check_count


'''
review_dir = walk_into_dir(movie_review_dir)
file_count = check_count = 0
for root_dir in review_dir:
    files = get_files(root_dir)
    tmp = check(root_dir, files)
    file_count += tmp[0]
    check_count += tmp[1]
print(file_count, check_count)
# 运行结果： 影评文件数： 59548   有影评没标签的电影数： 4325
'''

# 取所有电影的评分作为字典
def get_movie_rate_dict():
    movie_rate_dict = {}
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


movie_rate_dict = get_movie_rate_dict()

# 生成 59000 部电影的评分矩阵
def generate_all_movie_rates():
    movie_rate_pos_dict = {}
    movie_rates_matrix = np.zeros([count_all_movies, tags_set.__len__()], dtype=np.int32)
    y = np.zeros([count_all_movies, 1], dtype=np.float32)  # 评分预测值，根据下标取得分
    pos = 0
    for movie, rate in movie_rate_dict.items():
        y[pos] = rate
        movie_rate_pos_dict[movie] = pos
        movie_pos = movie_order_dict.get(movie, -1)
        if movie_pos > -1:
            movie_rates_matrix[pos] = user_item_matrix[movie_pos]
        else:
            pass
        pos += 1
    return movie_rates_matrix, y, movie_rate_pos_dict

movie_rates_matrix, movie_rates_y, movie_rate_pos_dict = generate_all_movie_rates()
# np.save(os.path.join(workdir, 'movie_rates_matrix.npy'), movie_rates_matrix)
# np.save(os.path.join(workdir, 'movie_rates_y.npy'), movie_rates_y)