""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Cong Le 
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
num_steps = 1500
batch_size = 64
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'w1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neutal_net(x):
    # layer 1
    layer_1 = tf.add(tf.matmul(x,weights['w1']), biases['b1'])
    #layer 2
    layer_2 = tf.add(tf.matmul(layer_1,weights['w2']), biases['b2'])
    # output full connected 
    out_layer = tf.add(tf.matmul(layer_2,weights['out']), biases['out'])

    return out_layer


# Construcct model

logits = neutal_net(X)
prediction = tf.nn.softmax(logits)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train_op = optimizer.minimize(loss_op)

#Evaluation model

correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables

init = tf.global_variables_initializer()

# Start training

with tf.Session()  as sess:
    sess.run(init)

    for step in range(1, num_steps +1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run opto,ization op
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step ==0 or step==1:
            #Calculate batch loss and accuracy 
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step" + str(step) + ", Loss=" + \
                "{:.4f}".format(loss) + ", Training accuracy=" + \
                "{:.4f}".format(acc))
    
    print("Optimizer finished ")

    # Cacilate accuracy for MNIST test images
    print("Testing accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))