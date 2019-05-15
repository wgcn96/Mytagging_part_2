
import tensorflow as tf
import numpy as np


from SVD.static import *


# 选取k个主成份，还原X矩阵
def reconstruct_using_svd(X, k):

    if k == 0:
        return X, 1., 0.

    (num_users, num_items) = X.shape

    graph = tf.Graph()
    with graph.as_default():
        # input arg
        R = tf.placeholder(tf.float32, shape=(num_users, num_items), name="R")

        # run SVD
        S, U, Vt = tf.svd(R, full_matrices=True)

        # reduce dimensions
        Sk = tf.diag(S)[0:k, 0:k]
        Uk = U[:, 0:k]
        Vk = tf.transpose(Vt)[0:k, :]

        # reconstruct matrix
        Rprime = tf.matmul(Uk, tf.matmul(Sk, Vk))

        # compute reconstruction RMSE
        rsquared = tf.linalg.norm(Rprime) / tf.linalg.norm(R)
        rmse = tf.metrics.root_mean_squared_error(R, Rprime)[1]

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        [Rprime_val, rsquared_val, rmse_val] = sess.run(
            [Rprime, rsquared, rmse], feed_dict={R: X})
        return Rprime_val, rsquared_val, rmse_val


# 按API默认形式完全还原X矩阵，k= min { num_user, num_item }
def reconstruct_using_svd_full(X):

    (num_users, num_items) = X.shape

    graph = tf.Graph()
    with graph.as_default():
        # input arg
        R = tf.placeholder(tf.float32, shape=(num_users, num_items), name="R")

        # run SVD
        # S, U, Vt = tf.svd(R, full_matrices=True)
        # Rprime = tf.matmul(U, tf.matmul(tf.linalg.diag(S), Vt, adjoint_b=True))

        s, u, v = tf.linalg.svd(X)
        Rprime = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))

        # compute reconstruction RMSE
        rsquared = tf.linalg.norm(Rprime) / tf.linalg.norm(R)
        rmse = tf.metrics.root_mean_squared_error(R, Rprime)[1]

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        [Rprime_val, rsquared_val, rmse_val] = sess.run(
            [Rprime, rsquared, rmse], feed_dict={R: X})
        return Rprime_val, rsquared_val, rmse_val


if __name__ == "__main__":
    rate_matrix = np.load(matrix_path).astype(np.float32)
    Rprime_val, rsquared_val, rmse_val = reconstruct_using_svd_full(rate_matrix)
    print("shape of reconstructed matrix: ", Rprime_val.shape)
