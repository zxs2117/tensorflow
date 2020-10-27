import tensorflow as tf

w=tf.Variable(.3,tf.float32)
b=tf.Variable(.3,tf.float32)

x=tf.placeholder(tf.float32)

LinerModel=w*x+b;
init=tf.global_variables_initializer()

with tf.Session() as sess:
 
  sess.run(init)
  print(sess.run(LinerModel,{x:[1,2,3,4]}))
