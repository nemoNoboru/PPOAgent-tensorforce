import tensorflow as tf
from tensorflow.contrib import layers


class NN:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(9))
        self.targetQ = tf.placeholder(tf.float32, shape=(5))
        features = [tf.reshape(self.input, [-1])]

        regularizer = layers.l2_regularizer(0.01)

        # Structure
        features = layers.fully_connected(features, 100, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 100, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 5, weights_regularizer=regularizer)

        self.predict = features[0]

        self.loss = tf.reduce_sum(tf.square(self.targetQ - self.predict))

        trainer = tf.train.AdamOptimizer()
        self.train = trainer.minimize(self.loss)
