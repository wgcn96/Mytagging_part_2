
import tensorflow as tf
import numpy as np
import os
from SVD.static import *



# p = tf.constant([[[3., 3., 4.]],
#                  [[3., 3., 4.]],
#                  [[3., 3., 4.]]], name='matrix1')
# p = tf.constant([[3.,  4.],
#                  [3.,  4.],
#                  [3.,  4.],
#                  [3.,  4.]], name='matrix1')
# p_C = tf.constant([[3.,  4.],
#                  [3.,  4.],
#                  [3.,  4.],
#                  [3.,  4.]], name='matrix2')
# p = tf.constant([3., 4.], name='matrix2')
# l2_p = tf.nn.l2_loss(p)
# l2_p = tf.sqrt(l2_p*2)
# l2_p = tf.multiply(p, p_C)
# l2_p = tf.reduce_sum(l2_p, 1)
# l2_p = tf.sqrt(l2_p)
# l2_p = tf.reduce_sum(l2_p, 0)
# sess = tf.Session()
# 
# print(sess.run(l2_p))



# matrix1 = tf.constant([[3., 3.]], name='matrix1')  # 1 row by 2 column
# matrix2 = tf.constant([[2.], [2.]], name='matrix2')  # 2 row by 1 column
# product = tf.matmul(matrix1, matrix2, name='product')

# sess = tf.Session()
#
# s, u, v = tf.svd(p)
# result_matrix = tf.matmul()
# init = tf.global_variables_initializer()
#
# sess.run(init)
# result = sess.run([s, u, v])
# print(result)


'''
k = 2
rate_matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1]])
rate_matrix_tensor = tf.convert_to_tensor(rate_matrix, dtype=tf.float32)
s, p, q = tf.svd(rate_matrix_tensor)
p = p[:, :k]
q = q[:, :k]
s = tf.linalg.diag(s)[:k, :k]
result_matrix = tf.matmul(p, tf.matmul(s, tf.transpose(q)))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
result = sess.run(result_matrix)
'''


# 两个矩阵相乘
x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
y = tf.constant([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]])
# 注意这里这里x,y要有相同的数据类型，不然就会因为数据类型不匹配而出错
z = tf.multiply(x, y)
z_sum = tf.reduce_sum(z, 1, keepdims=True)

# 两个数相乘
x1 = tf.constant(1)
y1 = tf.constant(2)
# 注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
z1 = tf.multiply(x1, y1)

# 数和矩阵相乘
x2 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
y2 = tf.constant(2.0)
cur_row = tf.constant([[0.], [1.], [2.]])
z5 = tf.pow(cur_row + 0.1, tf.constant(-0.5))
z6 = tf.multiply(z5, cur_row)
sum_row = tf.constant([0., 1., 2.])
# 注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
z2 = tf.multiply(x2, y2)
z3 = tf.multiply(cur_row, x2)
print(cur_row.shape.as_list())
z4 = tf.reduce_sum(sum_row)

with tf.Session() as sess:
    # print(sess.run(z))
    # print(sess.run(z_sum))
    print(sess.run(z2))
    print(sess.run(z3))
    print(sess.run(z5))
    print(sess.run(z6))
    # print(sess.run(z4))


'''
cur_row = tf.constant([0,2,1])
rate_matrix = np.array([[1,0,1],[0,1,0],[0,1,1]])
result = tf.nn.embedding_lookup(rate_matrix, cur_row)
with tf.Session() as sess:
    print(sess.run(result))
'''

# b_user = tf.Variable(1e-4 * tf.random.uniform([20, 1], 0, 1, dtype=tf.float32), name="b_user")
# init = tf.global_variables_initializer()
# sess = tf.Session()
# with sess.as_default():
#     writer = tf.summary.FileWriter(os.path.join(log_dir, "test1"), sess.graph)
#     sess.run(init)
#     tf.summary.histogram('histogram', b_user)
#     merged = tf.summary.merge_all()
#     rs = sess.run(merged)
#     writer.add_summary(rs, 0)
#
# print(b_user.eval(sess))
