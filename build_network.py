#coding:utf-8
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
"""
通常神经层都包括输入层、隐藏层和输出层。
这里的输入层只有一个属性， 所以我们就只有一个输入；
隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元；
输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。
所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。
"""
def add_layer(inputs,in_size,out_size,active_function):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))
    Wx_plsu_b = tf.matmul(inputs,Weights) + biases
    if active_function:
        outputs = active_function(Wx_plsu_b)
    else:
        outputs = Wx_plsu_b

#create data
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]   #默认是float64,生成300个值，增加维度变成300行一列
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise                        #生成一个y=x*2的数据

# plt.scatter(x_data, y_data)
# plt.show()

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
l1 = add_layer(xs,1,10,active_function=tf.nn.relu)
prediction = add_layer(l1,10,1,active_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))