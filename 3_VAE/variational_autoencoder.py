""" Variational Auto-Encoder Example.

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.
Using network with 3 hidden layer 

Author: Cong Le
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 784 # MNIST images are 28x28 pixels
hidden_dim_1 = 512
hidden_dim_2 = 256
hidden_dim_3 = 128
latent_dim = 2

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim_1])),
    'encoder_h2': tf.Variable(glorot_init([hidden_dim_1, hidden_dim_2])),
    'encoder_h3': tf.Variable(glorot_init([hidden_dim_2, hidden_dim_3])),
    'z_mean': tf.Variable(glorot_init([hidden_dim_3, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim_3, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim_3])),
    'decoder_h2': tf.Variable(glorot_init([hidden_dim_3, hidden_dim_2])),
    'decoder_h3': tf.Variable(glorot_init([hidden_dim_2, hidden_dim_1])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim_1, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim_1])),
    'encoder_b2': tf.Variable(glorot_init([hidden_dim_2])),
    'encoder_b3': tf.Variable(glorot_init([hidden_dim_3])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim_3])),
    'decoder_b2': tf.Variable(glorot_init([hidden_dim_2])),
    'decoder_b3': tf.Variable(glorot_init([hidden_dim_1])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Building the encoder
def vae_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    encoder_1 = tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1']
    encoder_1 = tf.nn.tanh(encoder_1)

    # Encoder Hidden layer with sigmoid activation #2
    encoder_2 = tf.matmul(encoder_1, weights['encoder_h2']) + biases['encoder_b2']
    encoder_2 = tf.nn.tanh(encoder_2)

    # Encoder Hidden layer with sigmoid activation #3
    encoder_3 = tf.matmul(encoder_2, weights['encoder_h3']) + biases['encoder_b3']
    encoder_3 = tf.nn.tanh(encoder_3)

    return encoder_3


# Building the decoder
def vae_decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    decoder_1 = tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1']
    decoder_1 = tf.nn.tanh(decoder_1)

    # Decoder Hidden layer with sigmoid activation #2
    decoder_2 = tf.matmul(decoder_1, weights['decoder_h2']) + biases['decoder_b2']
    decoder_2 = tf.nn.tanh(decoder_2)

    # Decoder Hidden layer with sigmoid activation #2
    decoder_3 = tf.matmul(decoder_2, weights['decoder_h3']) + biases['decoder_b3']
    decoder_3 = tf.nn.tanh(decoder_3)
    
    # Decoder Hidden layer with sigmoid activation #2
    decoder_out = tf.matmul(decoder_3, weights['decoder_out']) + biases['decoder_out']
    decoder_out = tf.nn.sigmoid(decoder_out)

    return decoder_out

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = vae_encoder(input_image)

# Calculate mean ans std 
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution of lattent space 
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps

# Building the decoder (with scope to re-use these layers later)

decoder = vae_decoder(z)

print(decoder)
# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, image_dim])

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Train
        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder, feed_dict={input_image: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.savefig("Original_VAE.png")

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig("Recon_image.png")