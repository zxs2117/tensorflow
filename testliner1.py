import tensorflow as tf
import numpy as np

#model parameters
w=tf.Variable(.3,tf.float32)
b=tf.Variable(-.3,tf.float32)

#model input&output
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

#model
out=w*x+b

#input_data
x_train=np.random.random_sample((100,)).astype(np.float32)
y_train=np.random.random_sample((100,)).astype(np.float32)

#loss function
loss=tf.reduce_sum(tf.square(out-y))

#optimizer
Optimizer=tf.train.GradientDescentOptimizer(0.001)
train=Optimizer.minimize(loss)

#training
init=tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})
    current_loss=sess.run(loss,{x:x_train,y:y_train})
    print("step %d training loss %f "% (i,current_loss))

  print(sess.run(w))
  print(sess.run(b))
