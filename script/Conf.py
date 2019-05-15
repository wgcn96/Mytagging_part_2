# -*- coding: UTF-8 -*-


import os
import configparser
"""
读取配置文件信息
"""


class ConfigParser:

    def __init__(self, file_path):
        self.config_dic = {}
        self.file_path = file_path

    def get_config(self):
        config = configparser.ConfigParser()
        config.read(self.file_path, encoding='utf8')
        for section in config.sections():
            print(section)
            for option in config.options(section):
                print(option)
                self.config_dic[option] = config.get(section, option)
        return self.config_dic


if __name__ == '__main__':
    print(os.path.realpath(__file__))
    print(os.getcwd())

    con = ConfigParser('settings.ini')
    res = con.get_config()
    print(res)
