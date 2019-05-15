#!/usr/bin/python
# coding:utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
global_step = tf.Variable(0, name='global_step', trainable=False)

y = []
z = []
N = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        # 阶梯型衰减
        learing_rate1 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=True)
        # 标准指数型衰减
        learing_rate2 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False)
        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        y.append(lr1[0])
        z.append(lr2[0])

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])
plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.title('exponential_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()


current_epoch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.03,
                                           current_epoch,
                                           decay_steps=num_epochs,
                                           decay_rate=0.03)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=current_epoch)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_epochs):
        current_epoch = i
