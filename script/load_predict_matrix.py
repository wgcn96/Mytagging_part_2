
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


def generate_implicit_matrix(content, tag_dict):
    matrix = np.zeros((59551, 2015), dtype=np.int32)

    all_movie_pos_file_Path = os.path.join(data_dir, "all_movie_matrix_dict.json")
    all_movie_pos_file = open(all_movie_pos_file_Path, encoding='utf-8', mode='r')
    all_movie_pos_dict = json.load(all_movie_pos_file)

    count = 0
    for key, cur_list in content.items():
        if len(cur_list) != 0:
            cur_movie_pos = all_movie_pos_dict.get(key)
            if cur_movie_pos is None:
                print(key)
                continue
            for tag in cur_list:
                tag_pos = tag_dict.get(tag, -1)
                if tag_pos > -1:
                    matrix[cur_movie_pos, tag_pos] = 1
                    count += 1

    print(np.sum(matrix), count)
    return matrix


if __name__ == "__main__":
    filePath = os.path.join(relative_data_dir, "predict_tag.json")
    content = loadJson(filePath)

    tagFile = os.path.join(data_dir, "tag.txt")
    tagList, tag_dict = loadTagList(tagFile)

    matrix = generate_implicit_matrix(content, tag_dict)

    result_file_path = os.path.join(data_dir, "predict_tag_matrix_ori.npy")
    np.save(result_file_path, matrix)

    ori_matrix = 2 * np.load(matrix_path)
    full_matrix = np.where(ori_matrix==0, matrix, ori_matrix)
    result_file_path = os.path.join(data_dir, "predict_tag_matrix.npy")
    np.save(result_file_path, full_matrix)
