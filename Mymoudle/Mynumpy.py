
import numpy as np
np.random.seed(0)
import os

from SVD.static import *

c_matrix = np.random.rand(20, 20)

if __name__ == '__main__':
    '''
    for i in range(20):
        pro = [0.1, 0, 0.3, 0.6, 0]
        a = np.random.choice(5, 1, p=pro)
        print(a)
        '''
    movie_list = [4,5,1,4]
    movie_list_log = np.log2(movie_list)
    print(movie_list_log)
    movie_list_log /= np.sum(movie_list_log)
    print(movie_list_log)
