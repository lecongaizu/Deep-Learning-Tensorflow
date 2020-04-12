""" Generative Adversarial Networks (GAN).

Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.
In Generator: Using AE with 2 hidden layer 
Author: COng Le
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 20000
batch_size = 128
learning_rate = 0.0001

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim_1 = 256
gen_hidden_dim_2 = 512
disc_hidden_dim_1 = 256
disc_hidden_dim_2 = 512
noise_dim = 100 # Noise data points

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim_1])),
    'gen_hidden2': tf.Variable(glorot_init([gen_hidden_dim_1, gen_hidden_dim_2])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim_2, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim_2])),
    'disc_hidden2': tf.Variable(glorot_init([disc_hidden_dim_2, disc_hidden_dim_1])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim_1, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim_1])),
    'gen_hidden2': tf.Variable(tf.zeros([gen_hidden_dim_2])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim_2])),
    'disc_hidden2': tf.Variable(tf.zeros([disc_hidden_dim_1])),
    'disc_out': tf.Variable(tf.zeros([1])),
}


# Generator
def generator(x):
    hidden_layer1 = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['gen_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)

    hidden_layer2 = tf.matmul(hidden_layer1, weights['gen_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['gen_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)


    out_layer = tf.matmul(hidden_layer2, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Discriminator
def discriminator(x):
    hidden_layer1 = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['disc_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)

    hidden_layer2 = tf.matmul(hidden_layer1, weights['disc_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['disc_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    out_layer = tf.matmul(hidden_layer2, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
            biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
d_loss = []
g_loss = []
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        g_loss.append(gl)
        d_loss.append(dl)
        
        if i % 1000 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
    epochs = range(1, num_steps+1)
    plt.plot(epochs, g_loss, 'g', label='Generator Loss')
    plt.plot(epochs, d_loss, 'b', label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')
    n = 4
    # Generate images from noise, using the generator network.
    img_gen = np.empty((28 * n, 28 * n))
    img_real = np.empty((28 * n, 28 * n))
    for i in range(n):
        # Noise input.
        batch_x, _ = mnist.test.next_batch(n)
        z = np.random.uniform(-1., 1., size=[n, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(n, 28, 28, 1))
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(n):
            # Draw the Generator digits
            img_gen[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])
            # a[j][i].imshow(img_gen)

        for j in range(n):
            # Draw the Generator digits
            img_real[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])

    
    # # f.show()
    # plt.draw()
    # print("Fake Images")
    # plt.imshow(img_gen, origin="upper", cmap="gray")
    # plt.savefig("Fake_Image.png")

    # print("Real Images")
    # plt.imshow(img_real, origin="upper", cmap="gray")
    # plt.savefig("Real_Image.png")

    


    


