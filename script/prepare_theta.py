
import json
import os
import collections
import numpy as np

from SVD.static import *


def loadJson(filePath):
    f = open(filePath, encoding='utf-8')
    content = json.load(f)
    return content


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


def generate_theta_matrix(content, tag_dict):
    matrix = np.zeros((2015, 2015), dtype=np.float32)

    count = 0
    for key_tag, item_dict in content.items():
        key_tag_pos = tag_dict.get(key_tag)
        if key_tag_pos is None:
            continue
        for i in range(10):
            cur_tag_list = item_dict.get(str(i))

            cur_tag_name = cur_tag_list[0]
            cur_tag_pos = tag_dict.get(cur_tag_name)
            if cur_tag_pos is not None:
                count += 1
                cur_tag_rela = cur_tag_list[1]
                matrix[key_tag_pos, cur_tag_pos] = cur_tag_rela

    print('total count: {}'.format(count))
    return matrix


def generate_theta_matrix_2(content, tag_dict):
    matrix = np.zeros((1506, 1506), dtype=np.float32)

    count = 0
    for key_tag, item_dict in content.items():
        key_tag_pos = tag_dict.get(key_tag)
        if key_tag_pos is None:
            continue
        for i in range(10):
            cur_tag_list = item_dict.get(str(i))

            cur_tag_name = cur_tag_list[0]
            cur_tag_pos = tag_dict.get(cur_tag_name)
            if cur_tag_pos is not None:
                count += 1
                cur_tag_rela = cur_tag_list[1]
                matrix[key_tag_pos, cur_tag_pos] = cur_tag_rela

    print('total count: {}'.format(count))
    return matrix



# 给18000千部电影做准备
def generate_theta_matrix_3(content, tag_dict):
    matrix = np.zeros((1896, 1896), dtype=np.float32)

    count = 0
    for key_tag, item_dict in content.items():
        key_tag_pos = tag_dict.get(key_tag)
        if key_tag_pos is None:
            continue
        for i in range(10):
            cur_tag_list = item_dict.get(str(i))

            cur_tag_name = cur_tag_list[0]
            cur_tag_pos = tag_dict.get(cur_tag_name)
            if cur_tag_pos is not None:
                count += 1
                cur_tag_rela = cur_tag_list[1]
                matrix[key_tag_pos, cur_tag_pos] = cur_tag_rela

    print('total count: {}'.format(count))
    return matrix


if __name__ == "__main__":
    filePath = os.path.join(relative_data_dir, "similar_tag.json")
    content = loadJson(filePath)

    tagFile = os.path.join(data3_dir, "tag.txt")
    tagList, tag_dict = loadTagList(tagFile)

    matrix = generate_theta_matrix_3(content, tag_dict)

    result_file_path = os.path.join(data3_dir, "tag_similar_matrix.npy")
    np.save(result_file_path, matrix)

    index_matrix = np.where(matrix != 0, np.ones(matrix.shape, np.float32), np.zeros(matrix.shape, np.float32))
    result_file_path = os.path.join(data3_dir, "tag_similar_index_matrix.npy")
    np.save(result_file_path, index_matrix)
